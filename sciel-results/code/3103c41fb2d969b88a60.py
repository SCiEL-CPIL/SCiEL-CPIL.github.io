import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _compute_block_sums(seq):
    """Compute all block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _build_conflict_graph(max_val, seq):
    """
    Build a conflict graph for potential next elements.
    Two candidate values x and y are in conflict if adding both (in order)
    would create a duplicate block sum.
    """
    n = len(seq)
    last = seq[-1] if seq else 0
    cur_sums = _compute_block_sums(seq)
    
    # Get all valid candidates
    valid_candidates = []
    for x in range(last + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            valid_candidates.append(x)
    
    # Build conflict graph
    # conflict[i][j] = True if adding candidates[i] then candidates[j] creates conflict
    k = len(valid_candidates)
    conflict = [[False] * k for _ in range(k)]
    
    for i in range(k):
        for j in range(k):
            if i == j:
                conflict[i][j] = True
                continue
            if i < j:
                # Add candidates[i] first, then candidates[j]
                seq1 = seq + [valid_candidates[i]]
                sums1 = _compute_block_sums(seq1)
                x2 = valid_candidates[j]
                tails2 = [x2]
                total = x2
                for a in reversed(seq1):
                    total += a
                    tails2.append(total)
                if sums1.intersection(tails2):
                    conflict[i][j] = True
                    conflict[j][i] = True
            else:
                # Already computed
                pass
    
    return valid_candidates, conflict

def _max_independent_set_tomita(n, conflict, deadline=None):
    """
    Find maximum independent set using Tomita's algorithm (branch and bound).
    conflict[i][j] = True means i and j CANNOT both be in the set.
    Returns the largest independent set as a list of indices.
    """
    # Tomita's algorithm works on the complement graph for maximum clique
    # We want maximum independent set, which is maximum clique in complement
    # But it's easier to think directly: we want max subset with no conflicts
    
    best_set = []
    
    # Precompute adjacency lists for conflict
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflict[i][j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to complement for maximum clique formulation
    # In complement: edge if NO conflict
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    # Tomita's maximum clique algorithm
    best_size = 0
    
    def expand(clique, cand):
        nonlocal best_size
        if deadline and time.time() > deadline:
            return
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_set = list(clique)
        
        if not cand:
            return
        
        # Order candidates by degree in complement
        ordered = sorted(cand, key=lambda v: -len(comp_adj[v]))
        
        for i in range(len(ordered)):
            v = ordered[i]
            if deadline and time.time() > deadline:
                return
            
            # Upper bound: current clique size + remaining candidates
            if len(clique) + len(ordered) - i <= best_size:
                break
            
            new_clique = clique + [v]
            new_cand = [u for u in ordered[i+1:] if u in comp_adj[v]]
            expand(new_clique, new_cand)
    
    expand([], list(range(n)))
    
    return best_set

def _solve_with_mis(seq, max_val=256, deadline=None):
    """
    Use Maximum Independent Set to find longest extension of seq.
    """
    if deadline and time.time() > deadline:
        return seq
    
    candidates, conflict = _build_conflict_graph(max_val, seq)
    if not candidates:
        return seq
    
    mis = _max_independent_set_tomita(len(candidates), conflict, deadline)
    
    # Reorder MIS by value (must be increasing)
    mis_vals = sorted([candidates[i] for i in mis])
    
    result = seq + mis_vals
    
    # Validate
    if _is_valid_full(result):
        return result
    
    # If MIS gives invalid result, fall back to greedy
    return _greedy_forward(seq, max_val)

def _beam_search(max_val, deadline, beam_width=100):
    """Beam search for longest valid sequence."""
    best_seq = []
    beam = [([], set(), 0)]  # (sequence, sums, last_val)
    
    while time.time() < deadline:
        next_beam = []
        
        for seq, sums, last in beam:
            for x in range(last + 1, max_val + 1):
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                if not sums.intersection(tails):
                    new_sums = sums | set(tails)
                    new_seq = seq + [x]
                    next_beam.append((new_seq, new_sums, x))
        
        if not next_beam:
            break
        
        # Update best
        for seq, _, _ in next_beam:
            if len(seq) > len(best_seq):
                best_seq = list(seq)
        
        # Prune beam by heuristic
        next_beam.sort(key=lambda s: (-len(s[0]), s[2], sum(s[0])))
        beam = next_beam[:beam_width]
        
        if all(s[2] >= max_val for s in beam):
            break
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = []
    best_len = 0
    
    # 1. Greedy for initial lower bound
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = list(greedy_seq)
        best_len = len(best)
        
    # 2. MIS-based approach
    if time.time() < deadline - 10:
        mis_seq = _solve_with_mis([], max_val=256, deadline=deadline - 10)
        if len(mis_seq) > best_len:
            best = list(mis_seq)
            best_len = len(best)
        
    # 3. Beam search with remaining time
    if time.time() < deadline - 5:
        beam_seq = _beam_search(max_val=256, deadline=deadline - 5, beam_width=200)
        if len(beam_seq) > best_len:
            best = list(beam_seq)
            best_len = len(best)
        
    # 4. If time remains, try beam with different width
    if time.time() < deadline - 3:
        beam_seq2 = _beam_search(max_val=256, deadline=deadline - 3, beam_width=500)
        if len(beam_seq2) > best_len:
            best = list(beam_seq2)
            best_len = len(best)
        
    # Final validation
    if not _is_valid_full(best):
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
