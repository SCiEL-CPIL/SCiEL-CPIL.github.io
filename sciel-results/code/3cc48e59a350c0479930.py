import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Greedy Set-Cover / Min-Conflict Local Search on the selection mask.
    - Instead of optimizing the permutation order, we optimize the final set of selected vertices.
    - However, the decoder is fixed: it picks vertices in descending score order.
    - Therefore, we need to find scores that induce a specific "good" ordering.
    - Insight: A good independent set is one where selected vertices have low degrees and few neighbors in the set.
    - We will construct a "target" independent set using a heuristic that maximizes size, 
      then derive scores that force the greedy decoder to pick exactly (or close to) this set.
    
    Algorithm:
    1. Find a high-quality independent set S using a local search on the mask itself.
       - Start with a greedy degree-based independent set.
       - Iterate: 
         a. Identify "bad" selected vertices: those that block many unselected vertices.
         b. Identify "good" unselected vertices: those that are blocked only by few selected vertices and have high potential.
         c. Swap: Remove a bad selected vertex v from S, add a good unselected vertex u to S if it becomes independent.
       - This is a 1-swap local search on the independent set mask.
    2. Once we have a target independent set S, assign scores to force the greedy decoder to pick S.
       - Vertices in S get high scores.
       - Vertices not in S get low scores.
       - Specifically, to ensure the greedy decoder picks S, we need that for any v in S, 
         all its neighbors u not in S must have lower priority than v, OR be blocked by other selected vertices.
       - A simpler robust strategy: 
         - Sort vertices in S by "criticality" (e.g., degree) to determine their relative order.
         - Assign scores such that vertices in S are picked first, in an order that avoids blocking each other (trivially true since they are independent).
         - Assign very low scores to non-S vertices.
       - To handle ties and ensure robustness, we can refine the scores within S based on a secondary heuristic (e.g., lower degree first) and assign non-S vertices scores interleaved or below.
       - Actually, the simplest way to force a specific independent set S to be picked by a greedy decoder is:
         - Assign score(v) = 1000 + rank_in_S(v) for v in S.
         - Assign score(v) = -1000 + rank_not_in_S(v) for v not in S.
         - This ensures all S vertices are picked before any non-S vertex.
         - Since S is independent, no vertex in S blocks another. So all S vertices will be picked.
         - Non-S vertices might be picked if they are not blocked by S. But we want to maximize the size.
         - Wait, if we just pick S, the size is |S|. The greedy decoder will pick all of S. 
           Will it pick any non-S vertices? 
           A non-S vertex u is picked if it is not a neighbor of any picked vertex.
           Since all S vertices are picked, u is picked iff u has no neighbors in S.
           So the final set is S ∪ {u not in S : N(u) ∩ S = ∅}.
           This is the maximal independent set containing S.
           So we want S to be a subset of a large maximal independent set.
           In fact, if we find a maximal independent set M, and set S=M, then the decoder picks M.
           So the goal is to find a large maximal independent set M.
    
    Revised Strategy:
    1. Find a large maximal independent set M using local search on the mask.
       - Start with greedy degree-based MIS.
       - Local search: try to swap a selected vertex v with an unselected vertex u to increase the size of the MIS.
         - If we remove v, some unselected vertices become unblocked.
         - If we add u, it must be independent of the remaining selected vertices.
         - Net change in size: -1 (remove v) + 1 (add u) + (number of new vertices that become unblocked and can be added).
         - This is complex. Simpler: just try to replace v with u if u is not adjacent to any other selected vertex (except v) and this allows adding more vertices later? 
         - Actually, a standard heuristic for MIS is:
           - While there is an improvement:
             - For each selected vertex v, calculate the number of unselected vertices that are ONLY blocked by v (i.e., N(u) ∩ S = {v}).
             - For each unselected vertex u, calculate the number of selected neighbors.
             - If we remove v, all u with N(u)∩S={v} become available. We can then greedily add them.
             - This is a 2-phase local search.
    
    Let's implement a simpler but effective local search:
    1. Initial MIS via greedy degree sorting.
    2. Iterate:
       - For each selected vertex v, compute "loss" = number of unselected vertices that would be freed if v were removed.
       - For each unselected vertex u, compute "gain" = 1 (if we could add it) but we can't just add it if it's blocked.
       - Alternative: Try all 1-swaps (remove v, add u).
         - A swap is valid if u is not adjacent to any selected vertex except v.
         - After swap, the set size remains the same. But we might be able to add more vertices.
         - So: perform the swap, then greedily add any unselected vertex that is now independent.
         - If the final size is larger, keep the swap.
    3. Once we have a good MIS M, assign scores to force the decoder to pick M.
       - Score(v) = 10000 - degree(v) for v in M. (Lower degree first within M, to be safe, though M is independent so order doesn't matter for blocking within M).
       - Score(v) = -10000 - degree(v) for v not in M.
       - This ensures M is picked first. Then any non-M vertex with no neighbors in M is picked.
       - The resulting set is M ∪ {u not in M : N(u) ∩ M = ∅}.
       - Since M is maximal, there are no such u. So the result is exactly M.
       - Thus, the size is |M|.
    
    This reduces the problem to finding a large maximal independent set.
    """
    start_time = time.time()
    deadline = 145.0

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbors
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    neighbor_sets = [set(neighbors_list[i]) for i in range(n)]
    deg = adj.sum(axis=1)

    # Helper: Greedily add any possible unselected vertices to the current set
    def expand_set(selected_mask):
        """
        Given a mask of selected vertices, greedily add any unselected vertex 
        that is not adjacent to any selected vertex.
        Returns the updated mask and the count.
        """
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if not selected_mask[u]:
                    # Check if u is adjacent to any selected vertex
                    is_adj = False
                    for v in neighbors_list[u]:
                        if selected_mask[v]:
                            is_adj = True
                            break
                    if not is_adj:
                        selected_mask[u] = True
                        changed = True
        return selected_mask, np.sum(selected_mask)

    # 1. Initial MIS
    order = np.argsort(deg, kind='stable')
    selected_mask = np.zeros(n, dtype=bool)
    for v in order:
        if not selected_mask[v]:
            # Check if v is adjacent to any selected vertex
            is_adj = False
            for u in neighbors_list[v]:
                if selected_mask[u]:
                    is_adj = True
                    break
            if not is_adj:
                selected_mask[v] = True
    
    best_mask = selected_mask.copy()
    best_size = np.sum(best_mask)
    
    # 2. Local Search: 1-swap + expand
    # We will try to swap a selected vertex v with an unselected vertex u.
    # To speed this up, we can limit the number of swaps per iteration or use a timeout.
    
    # Precompute for each unselected vertex u, the list of selected neighbors
    # This is dynamic, so we update it.
    
    # Let's do a few rounds of local search.
    # In each round, try all possible swaps (v, u) where v is selected, u is not, 
    # and u is not adjacent to any selected vertex except v.
    # If such a swap leads to a larger expanded set, perform it.
    
    max_rounds = 20
    for r in range(max_rounds):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        # Get current selected and unselected lists
        selected_list = np.where(selected_mask)[0]
        unselected_list = np.where(~selected_mask)[0]
        
        # For efficiency, we can just iterate through unselected vertices and check if they are "almost" independent.
        # i.e., their only selected neighbor is some v.
        
        # We'll try to find the best swap.
        best_swap_gain = 0
        best_v = -1
        best_u = -1
        
        # To make this faster, we can sample or limit the search.
        # With n=200, checking all pairs is O(n^2 * deg) which is 40000 * 20 = 800k ops per round.
        # 20 rounds = 16M ops. This is fine in Python if optimized.
        
        for u in unselected_list:
            # Find selected neighbors of u
            sel_neighbors = [v for v in neighbors_list[u] if selected_mask[v]]
            if len(sel_neighbors) == 1:
                v = sel_neighbors[0]
                # Try swapping v out and u in
                # Temporarily remove v, add u
                # Then expand
                # To avoid modifying the main mask, we can simulate the expansion on a copy.
                # But copying a 200-size mask is cheap.
                
                new_mask = selected_mask.copy()
                new_mask[v] = False
                new_mask[u] = True
                
                # Expand
                # We only need to expand from the vertices that might be affected.
                # Actually, just run the full expand_set.
                _, new_size = expand_set(new_mask)
                
                gain = new_size - best_size
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_v = v
                    best_u = u
                    best_new_mask = new_mask
                    # Note: best_new_mask is the mask AFTER the swap but BEFORE expansion?
                    # No, expand_set returns the expanded mask. We need to capture it.
                    # Let's re-capture the mask.
                    # The expand_set function modifies the mask in place and returns it.
                    # So best_new_mask should be the returned mask.
                    # Let's refactor expand_set to return the mask.
            
            # Optimization: If we find a gain > 0, we can break early? 
            # No, we want the best swap. But if time is tight, we can take the first improvement.
            # Given the budget, let's stick to finding the best swap in this round.
        
        if best_swap_gain > 0:
            selected_mask = best_new_mask
            best_size = best_swap_gain + best_size # Wait, best_size was the old size. 
            # Actually, new_size = best_size + gain. So best_size = new_size.
            # Let's just re-calculate best_size = np.sum(selected_mask) to be safe.
            best_size = np.sum(selected_mask)
            if best_size > np.sum(best_mask):
                best_mask = selected_mask.copy()
            improved = True
        
        if not improved:
            break

    # 3. Assign scores
    # Vertices in best_mask get high scores.
    # Vertices not in best_mask get low scores.
    # To ensure the greedy decoder picks best_mask, we assign:
    # score(v) = 10000 - deg[v] for v in best_mask
    # score(v) = -10000 - deg[v] for v not in best_mask
    
    scores = np.zeros(n)
    for v in range(n):
        if best_mask[v]:
            scores[v] = 10000.0 - float(deg[v])
        else:
            scores[v] = -10000.0 - float(deg[v])
    
    return scores
