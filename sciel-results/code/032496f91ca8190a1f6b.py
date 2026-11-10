# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def _build_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _is_qnr_mod_p(a, p):
    """Check if a is a quadratic non-residue mod p."""
    if a % p == 0:
        return False
    return pow(a, (p - 1) // 2, p) == p - 1


def _find_max_qnr_clique(p, time_budget=10):
    """
    Find the largest subset of Z_p such that all pairwise differences are QNRs.
    This is the max clique in the graph where edges connect QNR differences.
    """
    start_time = time.time()
    # Build adjacency matrix or list
    # adj[i] contains j if i-j is QNR
    adj = [set() for _ in range(p)]
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if _is_qnr_mod_p((i - j) % p, p):
                adj[i].add(j)
    
    best_clique = set()
    
    # Use a simple backtracking search for max clique
    # Given the small size (41), we can be exhaustive if needed, but use heuristics.
    
    def expand(clique, candidates):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = set(clique)
        if not candidates:
            return
        if time.time() - start_time > time_budget:
            return
            
        # Order candidates by degree in the candidate subgraph (greedy heuristic)
        # Or just pick one
        # To ensure we find max, we might need to try all.
        # But p=41 is small.
        
        # Pruning: if len(clique) + len(candidates) <= len(best_clique), return
        if len(clique) + len(candidates) <= len(best_clique):
            return

        # Iterate through candidates
        for v in list(candidates):
            new_clique = clique | {v}
            # New candidates must be neighbors of v in the current candidate set
            new_candidates = candidates.intersection(adj[v])
            expand(new_clique, new_candidates)
            candidates.remove(v)
            
    # Start with all nodes
    initial_candidates = set(range(p))
    # To improve speed, start with a greedy clique
    greedy_clique = set()
    remaining = set(range(p))
    while remaining:
        # Pick node with max degree in remaining
        best_node = max(remaining, key=lambda x: len(adj[x].intersection(remaining)))
        greedy_clique.add(best_node)
        remaining = remaining.intersection(adj[best_node])
        
    best_clique = greedy_clique
    # Now try to improve
    # We can run expand from empty set, but it might be slow.
    # Let's just run expand on the full set with a time limit.
    # Since p=41, the search space is manageable with good pruning.
    
    # Re-initialize for exhaustive-ish search with pruning
    best_clique = set()
    def expand_exhaustive(clique, candidates):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = set(clique)
        if not candidates:
            return
        if time.time() - start_time > time_budget:
            return
        if len(clique) + len(candidates) <= len(best_clique):
            return
        
        # Heuristic: try nodes with higher degree first
        sorted_cands = sorted(candidates, key=lambda x: -len(adj[x].intersection(candidates)))
        for v in sorted_cands:
            new_clique = clique | {v}
            new_candidates = candidates.intersection(adj[v])
            expand_exhaustive(new_clique, new_candidates)
            candidates.remove(v)
            
    expand_exhaustive(set(), set(range(p)))
    
    return best_clique


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 620
    
    forbidden = _build_forbidden(m)
    
    # Strategy: CRT decomposition
    # Z_205 ~ Z_5 x Z_41
    # We pick two residue classes mod 5, say 0 and 2.
    # For elements in these classes, the mod 5 difference is 0 (within class) or 2/3 (between classes).
    # If diff mod 5 is 2 or 3, the pair is safe regardless of mod 41.
    # If diff mod 5 is 0, we need the mod 41 difference to be a QNR.
    
    # So we need a set S_41 subset Z_41 such that all pairwise differences are QNRs.
    # Let K be the max size of such a set.
    # We can pick K elements in G_0 (x % 5 == 0) and K elements in G_2 (x % 5 == 2).
    # Total size 2K.
    
    s_41 = _find_max_qnr_clique(41, time_budget=5)
    k = len(s_41)
    
    # Construct the set
    # Elements in G_0: x such that x % 5 == 0 and x % 41 in s_41
    # Elements in G_2: x such that x % 5 == 2 and x % 41 in s_41
    
    # We need to map (a5, a41) to x mod 205.
    # x = a5 * 41 * inv(41, 5) + a41 * 5 * inv(5, 41) mod 205
    # 41 = 1 mod 5. inv(1, 5) = 1.
    # 5 mod 41 = 5. inv(5, 41): 5 * 33 = 165 = 4*41 + 1. So inv is 33.
    
    inv_41_mod_5 = pow(41, -1, 5) # 1
    inv_5_mod_41 = pow(5, -1, 41) # 33
    
    def crt_combine(a5, a41):
        # x = a5 * 41 * 1 + a41 * 5 * 33
        return (a5 * 41 + a41 * 165) % 205
    
    result_set = set()
    
    # Add elements from G_0
    for a41 in s_41:
        x = crt_combine(0, a41)
        result_set.add(x)
        
    # Add elements from G_2
    for a41 in s_41:
        x = crt_combine(2, a41)
        result_set.add(x)
        
    # Verify validity
    def is_valid(s_set):
        s_list = list(s_set)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i], s_list[j]
                d1 = (a - b) % m
                d2 = (b - a) % m
                if d1 in forbidden or d2 in forbidden:
                    return False
        return True

    if not is_valid(result_set):
        # Fallback to greedy if something went wrong
        # (Should not happen if logic is correct)
        # For now, return empty or a known small set
        return np.array([0], dtype=np.int64)

    return np.array(sorted(list(result_set)), dtype=np.int64)


# EVOLVE-BLOCK-END
