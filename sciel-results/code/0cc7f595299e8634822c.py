"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Generate all candidate pairs (m, r) for m in [20, 200].
    # 2. Build a conflict graph where an edge exists between (m1, r1) and (m2, r2)
    #    if they are NOT disjoint, i.e., r1 % gcd(m1, m2) == r2 % gcd(m1, m2).
    # 3. Find a Maximum Weight Independent Set (MWIS) in this graph, where weight is 1/m.
    #    Since MWIS is NP-hard, we use a greedy heuristic:
    #    - Sort candidates by weight (1/m) descending.
    #    - Iterate through candidates, adding to the solution set if it doesn't conflict
    #      with any already selected candidate.
    #    - To improve the greedy result, we can try multiple random orderings or 
    #      a simple local search (swap) if time permits.
    #    - Given the structure, a simple greedy might be suboptimal, but it's fast.
    #    - A better greedy: Sort by m ascending (larger weight first).
    #      However, conflicts depend on residues. 
    #      Let's try a "Greedy with Backtracking" or "Local Search".
    #      Actually, a simple greedy is O(N^2) which is ~ 180*200 * 180*200 / 2? No.
    #      Number of candidates N = sum_{m=20}^{200} m = ~18000.
    #      Checking conflicts against selected set is O(K) per candidate, where K is size of selected set.
    #      Total O(N * K). K is likely small (maybe < 100). N*K ~ 1.8M operations. Very fast.
    #      The bottleneck is building the conflict check.
    #      We can precompute nothing, just check on the fly.
    
    # Let's refine the greedy:
    # 1. List all candidates (m, r).
    # 2. Sort by m ascending (so 1/m is descending).
    # 3. Initialize empty selected list.
    # 4. For each candidate (m, r) in sorted order:
    #    Check if (m, r) is disjoint from all (m', r') in selected.
    #    If yes, add to selected.
    
    # This greedy is deterministic. To potentially find a better set, we can randomize the order of residues for a given m, or randomize the tie-breaking.
    # Or, we can try to add candidates in a different order.
    # Let's try a few variations of the greedy order to see if we can beat the simple one.
    
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r, 1.0/m))
    
    # Sort by weight descending (m ascending)
    candidates.sort(key=lambda x: x[0])
    
    best_selected = []
    best_score = 0.0
    
    # Run greedy multiple times with different random shuffles of residues for same m?
    # Or just run the standard greedy.
    # Standard Greedy:
    selected = []
    for m, r, w in candidates:
        # Check conflict with all in selected
        is_conflict = False
        for m_sel, r_sel, _ in selected:
            g = gcd(m, m_sel)
            if r % g == r_sel % g:
                is_conflict = True
                break
        if not is_conflict:
            selected.append((m, r, w))
    
    if selected:
        score = sum(w for _, _, w in selected)
        if score > best_score:
            best_score = score
            best_selected = selected[:]
    
    # Try a slightly smarter approach:
    # The simple greedy might fail to pick a small m because it conflicts with an even smaller m that was picked earlier,
    # but maybe swapping would allow more total weight?
    # Local Search:
    # 1. Start with greedy solution.
    # 2. Try to add any candidate not in solution if it doesn't conflict. (Greedy already did this in order, so none should be addable without conflict with existing? 
    #    Wait, greedy processes in order. If a candidate was skipped, it conflicted with some selected item. 
    #    If we remove a selected item, it might become addable. But that's complex.)
    
    # Alternative: Randomized Greedy
    # Shuffle the candidates within each modulus group? No, m is fixed.
    # Shuffle the order of checking?
    # Let's try a few random restarts of the greedy with randomized residue order for each m.
    
    import random
    
    for _ in range(5): # 5 random restarts
        if time.time() > deadline - 10:
            break
            
        # Create a list of candidates, but for each m, shuffle the residues
        # This effectively randomizes which residue we pick for a given m if we use a "first fit" logic?
        # No, the greedy logic is "if it fits, take it".
        # If we shuffle the global list, we might pick a large m first, blocking small ones.
        # We want to prioritize small m.
        # So we keep m order, but for each m, we can try different residues?
        # The standard greedy picks the FIRST valid residue for a given m (since r is 0..m-1).
        # If we shuffle r order, we might pick a different residue, allowing more future picks.
        
        shuffled_candidates = []
        for m in range(20, 201):
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                shuffled_candidates.append((m, r, 1.0/m))
        
        selected = []
        for m, r, w in shuffled_candidates:
            is_conflict = False
            for m_sel, r_sel, _ in selected:
                g = gcd(m, m_sel)
                if r % g == r_sel % g:
                    is_conflict = True
                    break
            if not is_conflict:
                selected.append((m, r, w))
        
        if selected:
            score = sum(w for _, _, w in selected)
            if score > best_score:
                best_score = score
                best_selected = selected[:]
    
    # One more strategy: Greedy by "Degree"
    # Nodes with fewer conflicts are easier to place.
    # But we want high weight.
    
    # Let's verify the best solution found
    if not best_selected:
        # Fallback to seed
        best_selected = [[20 * j, j - 1, 1.0/(20*j)] for j in range(1, 11)]
        best_score = sum(1.0/(20*j) for j in range(1, 11))
    
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r, _ in best_selected]

# EVOLVE-BLOCK-END
