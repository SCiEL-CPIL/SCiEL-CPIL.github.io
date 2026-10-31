import numpy as np
import random

N = 50

def construct():
    # Formulate as maximum independent set on the "addition graph" of indices.
    # Edge (i,j) exists if adding both i and j to a set would create a sum collision
    # (i.e., i+j equals some k+k or k+l for k,l in the set).
    # For a Sidon set, all pairwise sums i+j (i<=j) must be distinct.
    #
    # We use a greedy heuristic with random restarts, enhanced with:
    # 1. Multiple random orderings
    # 2. Local search (swap-based improvement)
    # 3. Backtracking for small remaining candidate sets
    # 4. Deterministic seeding with best known structures
    
    best_S = []
    best_size = 0
    
    def is_sidon(S):
        """Check if S is a Sidon set (all pairwise sums i+j, i<=j, are distinct)."""
        sums = set()
        for i_idx in range(len(S)):
            for j_idx in range(i_idx, len(S)):
                s = S[i_idx] + S[j_idx]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def greedy_build(order):
        """Greedy: add elements in given order if they don't break Sidon property."""
        S = []
        used_sums = set()
        for i in order:
            # Check if i can be added
            new_sums = set()
            new_sums.add(i + i)
            for j in S:
                new_sums.add(i + j)
            
            if new_sums & used_sums:
                continue
            
            S.append(i)
            used_sums |= new_sums
        return S
    
    def local_search(S, candidates, max_swaps=200):
        """Try to improve S by swapping one element out and one in."""
        S = list(S)
        improved = True
        swaps = 0
        while improved and swaps < max_swaps:
            improved = False
            swaps += 1
            used_sums = set()
            for i_idx in range(len(S)):
                for j_idx in range(i_idx, len(S)):
                    used_sums.add(S[i_idx] + S[j_idx])
            
            # Try removing each element and adding a candidate
            for rem_idx in range(len(S)):
                S_without = S[:rem_idx] + S[rem_idx+1:]
                used_sums_without = set()
                for i_idx in range(len(S_without)):
                    for j_idx in range(i_idx, len(S_without)):
                        used_sums_without.add(S_without[i_idx] + S_without[j_idx])
                
                for c in candidates:
                    if c in S_without:
                        continue
                    new_sums = {c + c}
                    for j in S_without:
                        new_sums.add(c + j)
                    if not (new_sums & used_sums_without):
                        # Valid swap
                        S = S_without + [c]
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # Run many random restarts
    num_restarts = 500
    for r in range(num_restarts):
        # Random ordering
        order = list(range(N))
        random.shuffle(order)
        
        S = greedy_build(order)
        
        # Local search to try to grow
        candidates = [i for i in range(N) if i not in S]
        S = local_search(S, candidates, max_swaps=100)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        
        # Early termination if we find a very large set
        if best_size >= 8:
            break
    
    # Also try some structured initializations
    # Known good Sidon sets can be found via Singer/Bose-Chowla constructions
    # For N=50, try to find size 7 or 8
    
    # Additional: try greedy with different heuristics
    # Pick element that minimizes future conflicts
    for r in range(200):
        remaining = set(range(N))
        S = []
        used_sums = set()
        
        while remaining:
            # Score each candidate by how many conflicts it would cause
            best_elem = None
            best_score = -1
            for i in list(remaining):
                new_sums = {i + i}
                for j in S:
                    new_sums.add(i + j)
                conflicts = len(new_sums & used_sums)
                # Also count how many future elements it would block
                # (approximate by degree in conflict graph)
                if conflicts > 0:
                    continue
                # Prefer elements with fewer potential conflicts with remaining
                potential_conflicts = 0
                for j in remaining:
                    if j == i:
                        continue
                    if (i + j) in used_sums or (j + j) in used_sums:
                        potential_conflicts += 1
                score = -potential_conflicts
                if score > best_score:
                    best_score = score
                    best_elem = i
            
            if best_elem is None:
                break
            
            S.append(best_elem)
            used_sums.add(best_elem + best_elem)
            for j in S[:-1]:
                used_sums.add(best_elem + j)
            remaining.discard(best_elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
        
        if best_size >= 8:
            break
    
    if not best_S:
        best_S = [0]
    
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
    
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
