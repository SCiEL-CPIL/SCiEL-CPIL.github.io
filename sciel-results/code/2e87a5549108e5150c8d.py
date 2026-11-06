import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

class Hypergraph:
    def __init__(self):
        self.presence = 0  # 560-bit integer
        self.common = [[0] * V for _ in range(V)]  # 16x16 matrix of 16-bit masks
        self.count = 0

    def add(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence |= (1 << idx)
        self.count += 1
        # Update common masks
        # {a,b,c} means c is in common[a][b], etc.
        self.common[a][b] |= (1 << c)
        self.common[b][a] |= (1 << c)
        self.common[a][c] |= (1 << b)
        self.common[c][a] |= (1 << b)
        self.common[b][c] |= (1 << a)
        self.common[c][b] |= (1 << a)

    def remove(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence &= ~(1 << idx)
        self.count -= 1
        # Update common masks
        self.common[a][b] &= ~(1 << c)
        self.common[b][a] &= ~(1 << c)
        self.common[a][c] &= ~(1 << b)
        self.common[c][a] &= ~(1 << b)
        self.common[b][c] &= ~(1 << a)
        self.common[c][b] &= ~(1 << a)

    def is_present(self, idx):
        return (self.presence >> idx) & 1

    def check_conflict(self, a, b, c):
        """
        Check if adding {a,b,c} creates a K(2,2,2).
        We assume {a,b,c} is not currently present.
        """
        # Candidates for a' must be in common[b][c] and not in {a,b,c}
        mask_bc = self.common[b][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        
        # Candidates for b' must be in common[a][c] and not in {a,b,c}
        mask_ac = self.common[a][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        
        # Candidates for c' must be in common[a][b] and not in {a,b,c}
        mask_ab = self.common[a][b] & ~(1 << a) & ~(1 << b) & ~(1 << c)

        # Iterate over possible a'
        a_p_mask = mask_bc
        while a_p_mask:
            a_p_bit = a_p_mask & (-a_p_mask)
            a_p = a_p_bit.bit_length() - 1
            a_p_mask ^= a_p_bit
            
            # For this a', iterate over possible b'
            # b' must be in mask_ac and not equal to a_p
            b_p_mask = mask_ac & ~(1 << a_p)
            while b_p_mask:
                b_p_bit = b_p_mask & (-b_p_mask)
                b_p = b_p_bit.bit_length() - 1
                b_p_mask ^= b_p_bit
                
                # Check if {a', b', c} is present
                # i.e., c is in common[a_p][b_p]
                if not (self.common[a_p][b_p] & (1 << c)):
                    continue
                
                # If {a', b', c} is present, we need to check if there exists c'
                # such that:
                # {a, b, c'} in H  -> c' in common[a][b]
                # {a, b', c'} in H -> c' in common[a][b_p]
                # {a', b, c'} in H -> c' in common[a_p][b]
                # {a', b', c'} in H -> c' in common[a_p][b_p]
                # and c' not in {a, b, c, a', b'}
                
                c_cand = (self.common[a][b] & 
                          self.common[a][b_p] & 
                          self.common[a_p][b] & 
                          self.common[a_p][b_p])
                
                # Exclude a, b, c, a', b'
                c_cand &= ~(1 << a)
                c_cand &= ~(1 << b)
                c_cand &= ~(1 << c)
                c_cand &= ~(1 << a_p)
                c_cand &= ~(1 << b_p)
                
                if c_cand:
                    return True
        
        return False

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_hg = None
    best_count = 0
    
    while time.time() < deadline:
        # Initialize with a greedy random order
        hg = Hypergraph()
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 10: # Reserve time for local search
                break
            a, b, c = ALL_TRIPLES[idx]
            if not hg.check_conflict(a, b, c):
                hg.add(idx)
        
        # Local Search
        # We will try to add missing triples or perform swaps.
        # To make it efficient, we keep a list of missing triples.
        present_mask = hg.presence
        missing = [i for i in range(N_TRIPLES) if not (present_mask >> i) & 1]
        
        improved = True
        stall_count = 0
        max_stall = 100 # Restart local search if stuck too long
        
        while improved and time.time() < deadline - 5:
            improved = False
            if stall_count > max_stall:
                break
            
            random.shuffle(missing)
            
            # Phase 1: Try to add
            for idx in missing:
                if time.time() > deadline - 5:
                    break
                a, b, c = ALL_TRIPLES[idx]
                if not hg.check_conflict(a, b, c):
                    hg.add(idx)
                    present_mask |= (1 << idx)
                    missing.remove(idx)
                    improved = True
                    stall_count = 0
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps (remove 1, add 1)
            # This is more complex. We need to find a present triple whose removal allows an addition.
            # Heuristic: Pick a random missing triple. Find a present triple that "blocks" it.
            # A present triple t blocks t_new if t is part of a K(2,2,2) with t_new.
            # Instead of finding blockers, just try random swaps.
            
            if missing:
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    if not missing:
                        break
                    new_idx = random.choice(missing)
                    # Pick a random present triple
                    # We don't have a direct list of present indices, but we can pick a random idx and check
                    # Or we can maintain a list of present indices.
                    # Let's maintain a list of present indices for faster sampling.
                    pass # We'll implement a better swap logic below if needed, but for now, let's see if greedy+add is enough.
                    # Actually, simple add-only local search might get stuck.
                    # Let's implement a simple 2-swap: remove t1, add t2.
                    # To do this efficiently, we need to know which present triples to consider.
                    # For now, let's just restart if we get stuck.
            
            if not improved:
                stall_count += 1
        
        if hg.count > best_count:
            best_count = hg.count
            best_hg = hg
        
        if best_count > 300: # Heuristic: if we're doing well, maybe we can stop early? No, keep searching.
            pass

    if best_hg is None:
        return []
        
    result = []
    for i in range(N_TRIPLES):
        if (best_hg.presence >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
