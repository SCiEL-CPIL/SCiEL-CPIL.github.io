import time
import random
import heapq

N = 256
MAX_SUM = 2 * N

def _is_sidon(selected):
    """Check if selected is a Sidon set using bitset for speed."""
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        # Check 2a
        if (sums >> (2 * a)) & 1:
            return False
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            if (sums >> s) & 1:
                return False
            sums |= (1 << s)
    return True

def _compute_sums_bitset(selected):
    sums = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            sums |= (1 << s)
    return sums

def _is_maximal_bitset(selected, sums_mask):
    """
    Check if selected is inclusion-maximal.
    Sums_mask represents all sums a+b for a<=b in selected.
    x is addable if:
    1. 2x is NOT in sums_mask
    2. For all a in selected, x+a is NOT in sums_mask
    """
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check diagonal
        if (sums_mask >> (2 * x)) & 1:
            continue
        # Check off-diagonal
        blocked = False
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                blocked = True
                break
        if not blocked:
            return False
    return True

class _BitTracker:
    __slots__ = ("sums", "sel_set", "members", "size")

    def __init__(self):
        self.sums = 0
        self.sel_set = set()
        self.members = []
        self.size = 0

    def add(self, x):
        # Check validity before adding? Caller should ensure.
        # But we assume caller checks Sidon property.
        self.sums |= (1 << (2 * x))
        for a in self.members:
            self.sums |= (1 << (x + a))
        self.members.append(x)
        self.sel_set.add(x)
        self.size += 1

    def remove(self, x):
        # Rebuilding sums is safer and fast enough for N=256 if done infrequently,
        # but let's try to be smart. 
        # Actually, removing from bitset is hard because sums might be shared.
        # e.g., 1+4=5 and 2+3=5. If we remove 1, 5 might still be in sums via 2+3.
        # So we MUST recalculate sums upon removal or maintain counts.
        # For simplicity and correctness in a tight loop, we will recalculate sums 
        # after a removal in the reduction step if we accept it.
        # For the tracker itself, we will just mark as removed and rely on 
        # external validation or recalculation.
        
        # To make this work efficiently, we will NOT maintain sums incrementally 
        # on removal in this class if it's ambiguous. 
        # Instead, the reduction algorithm will:
        # 1. Remove x from members/sel_set.
        # 2. Recalculate sums_mask from scratch (O(k^2), k~20 -> 400 ops).
        # 3. Check maximality.
        # 4. If fail, restore x and restore old sums_mask.
        
        self.sel_set.discard(x)
        self.members.remove(x)
        self.size -= 1
        # Note: self.sums is now STALE. 
        # The caller must handle sum recalculation if they want to use self.sums.
        # We will provide a method to recalc.

    def recalc_sums(self):
        self.sums = 0
        for i in range(len(self.members)):
            a = self.members[i]
            self.sums |= (1 << (2 * a))
            for j in range(i + 1, len(self.members)):
                s = a + self.members[j]
                self.sums |= (1 << s)

def _greedy_bitset(order, deadline):
    """
    Greedy construction using bitset for fast checks.
    order: list of candidates to try.
    """
    selected = []
    sums = 0
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        if (sums >> (2 * x)) & 1:
            continue
        ok = True
        for a in selected:
            if (sums >> (x + a)) & 1:
                ok = False
                break
        if ok:
            selected.append(x)
            sums |= (1 << (2 * x))
            for a in selected[:-1]:
                sums |= (1 << (x + a))
    return selected

def _reduce_bitset(start_set, deadline):
    """
    Aggressively reduce the set by removing elements that don't break maximality.
    Uses bitset for checks.
    """
    current = list(start_set)
    if not current:
        return current
    
    # Initial sums
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        # Shuffle to avoid bias? Or try all?
        # Trying all is better for finding the minimal set.
        for i in range(len(current)):
            if time.time() > deadline:
                break
            x = current[i]
            # Remove x
            new_current = current[:i] + current[i+1:]
            if not new_current:
                # Empty set is not maximal (unless N=0, but N=256)
                continue
            
            # Recalculate sums for new_current
            new_sums = _compute_sums_bitset(new_current)
            
            # Check maximality
            if _is_maximal_bitset(new_current, new_sums):
                # Accept removal
                current = new_current
                sums_mask = new_sums
                improved = True
                break # Restart loop with new size
    
    return current

def _swap_improve_bitset(start_set, deadline, rng):
    """
    Try swaps to find better configurations that allow further reduction.
    """
    current = list(start_set)
    current_set = set(current)
    sums_mask = _compute_sums_bitset(current)
    
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 100:
        improved = False
        iters += 1
        
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N + 1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            # Remove r
            temp_current = [x for x in current if x != r]
            temp_set = set(temp_current)
            
            for a in add_candidates:
                if time.time() > deadline:
                    break
                if a in temp_set:
                    continue
                
                # Check if temp_current + [a] is Sidon
                # We can check by seeing if a conflicts with temp_current
                # Sums of temp_current are known? No, we didn't calc them.
                # Let's just build the candidate and check Sidon via bitset.
                candidate = temp_current + [a]
                candidate.sort()
                
                if not _is_sidon(candidate):
                    continue
                
                # Check maximality
                cand_sums = _compute_sums_bitset(candidate)
                if _is_maximal_bitset(candidate, cand_sums):
                    # Accept swap
                    current = candidate
                    current_set = set(candidate)
                    sums_mask = cand_sums
                    improved = True
                    break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best = None
    best_size = N + 1
    
    candidates = []
    
    # 1. Greedy Ascending
    c1 = _greedy_bitset(list(range(1, N + 1)), deadline)
    candidates.append(c1)
    
    # 2. Greedy Descending
    c2 = _greedy_bitset(list(range(N, 0, -1)), deadline)
    candidates.append(c2)
    
    # 3. Randomized Greedy
    rng_master = random.Random(42)
    for seed in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng = random.Random(1000 + seed)
        rng.shuffle(order)
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)
    
    # 4. Sparsity-focused Greedy (skip some)
    for seed in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(2000 + seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # Filter some out randomly to encourage sparser starts? 
        # Actually, just using different shuffles is usually enough.
        sel = _greedy_bitset(order, deadline)
        if sel:
            candidates.append(sel)

    # Process candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if not cand:
            continue
        
        # Ensure valid Sidon (greedy should ensure this, but just in case)
        if not _is_sidon(cand):
            continue
            
        # Check if maximal. If not, try to make it maximal? 
        # Greedy produces maximal sets by construction (it adds if possible, 
        # so at the end, nothing else can be added).
        # However, if we broke early due to deadline, it might not be maximal.
        # We should only process if it's maximal, or try to complete it.
        # For now, skip non-maximal.
        sums = _compute_sums_bitset(cand)
        if not _is_maximal_bitset(cand, sums):
            # Try to complete it quickly?
            # Or just skip.
            continue
            
        rng = random.Random(hash(tuple(sorted(cand))) % (2 ** 31))
        
        # Reduce
        minimized = _reduce_bitset(cand, deadline)
        
        # Swap Improve
        if time.time() < deadline - 5.0:
            minimized = _swap_improve_bitset(minimized, deadline, rng)
            # Reduce again after swaps
            minimized = _reduce_bitset(minimized, deadline)
            
        if minimized:
            if _is_sidon(minimized) and _is_maximal_bitset(minimized, _compute_sums_bitset(minimized)):
                if len(minimized) < best_size:
                    best_size = len(minimized)
                    best = list(minimized)

    if best is None:
        # Fallback
        return _greedy_bitset(list(range(1, N + 1)), start_time + 1.0)
        
    return best
