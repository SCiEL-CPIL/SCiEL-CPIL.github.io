import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m_target = 401
    
    # Bitmask representation of adjacency matrix
    # adj[i] is an integer where bit j is 1 if edge (i, j) exists.
    
    def compute_books(adj):
        """
        Returns:
        max_book: integer, the maximum number of triangles containing any edge.
        sum_sq_books: integer, sum of squares of book sizes for all edges.
                       This is a secondary objective to flatten the distribution.
        """
        max_book = 0
        sum_sq = 0
        for i in range(n):
            mask_i = adj[i]
            # Only check edges (i, j) where j > i to avoid duplicates
            for j in range(i + 1, n):
                if (mask_i >> j) & 1:
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book:
                        max_book = common
                    sum_sq += common * common
        return max_book, sum_sq

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def edge_count(adj):
        count = 0
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            count += mask.bit_count()
        return count

    def random_swap(adj, rng):
        """
        Performs a random edge swap: removes (u,v) and adds (x,y).
        Returns True if swap was performed, False if failed to find valid edges.
        """
        # Find a random edge
        edge_found = False
        for _ in range(100):
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u != v and ((adj[u] >> v) & 1):
                edge_found = True
                break
        if not edge_found:
            return False
            
        # Find a random non-edge
        non_edge_found = False
        for _ in range(100):
            x = rng.randint(0, n-1)
            y = rng.randint(0, n-1)
            if x != y and not ((adj[x] >> y) & 1):
                # Ensure we are not swapping the same edge
                if (u == x and v == y) or (u == y and v == x):
                    continue
                non_edge_found = True
                break
        if not non_edge_found:
            return False

        # Perform swap
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        return True

    def greedy_local_search(adj, max_time, rng):
        """
        Strict greedy descent on (max_book, sum_sq_books).
        If stuck, performs a random kick.
        """
        cur_max, cur_sum_sq = compute_books(adj)
        best_max, best_sum_sq = cur_max, cur_sum_sq
        best_adj = adj.copy()
        
        stuck_count = 0
        kick_threshold = 500 # Number of failed improvements before a kick
        kick_size = 5        # Number of random swaps in a kick
        
        while time.time() < max_time:
            # Try to find an improving move
            improved = False
            # To speed up, we can limit the number of random tries per iteration
            # but for n=40, full scan is too slow. We rely on random sampling of swaps.
            
            # Sample a random swap
            if random_swap(adj, rng):
                new_max, new_sum_sq = compute_books(adj)
                
                # Acceptance criteria:
                # 1. If new_max < cur_max: Accept (improvement in primary)
                # 2. If new_max == cur_max and new_sum_sq < cur_sum_sq: Accept (improvement in secondary)
                # 3. Else: Revert
                if new_max < cur_max or (new_max == cur_max and new_sum_sq < cur_sum_sq):
                    cur_max, cur_sum_sq = new_max, new_sum_sq
                    improved = True
                    stuck_count = 0
                    if cur_max < best_max or (cur_max == best_max and cur_sum_sq < best_sum_sq):
                        best_max, best_sum_sq = cur_max, cur_sum_sq
                        best_adj = adj.copy()
                        if best_max == 7: # Theoretical lower bound for this size
                            return best_adj, best_max, best_sum_sq
                else:
                    # Revert the swap
                    # We need to know which edges were swapped to revert.
                    # Since random_swap doesn't return the edges, this is a problem.
                    # Let's modify random_swap to return the edges or do the swap logic inline.
                    # For now, let's assume we can recompute or we need a better structure.
                    # Actually, reverting requires knowing u,v,x,y.
                    # Let's refactor the loop to handle revert correctly.
                    pass # Placeholder, logic below will be more robust
                    
            # The above logic is flawed because random_swap doesn't return info to revert.
            # Let's rewrite the inner loop to be self-contained for swap/revert.
            
            # Re-implementation of the inner loop for correctness
            if random_swap(adj, rng):
                # We need to know what was swapped. 
                # Since I can't easily modify random_swap to return u,v,x,y without changing signature,
                # let's just do the swap logic inline here to allow reverting.
                pass

        return best_adj, best_max, best_sum_sq

    # Refactored approach: Inline swap logic to allow reverting
    def run_greedy_search(adj, max_time, rng):
        cur_max, cur_sum_sq = compute_books(adj)
        best_max, best_sum_sq = cur_max, cur_sum_sq
        best_adj = adj.copy()
        
        stuck_count = 0
        kick_threshold = 200
        kick_size = 3
        
        while time.time() < max_time:
            # Attempt a random swap
            u = v = x = y = -1
            swap_valid = False
            for _ in range(50): # Try up to 50 times to find valid edge and non-edge
                u = rng.randint(0, n-1)
                v = rng.randint(0, n-1)
                if u == v or not ((adj[u] >> v) & 1):
                    continue
                x = rng.randint(0, n-1)
                y = rng.randint(0, n-1)
                if x == y or ((adj[x] >> y) & 1):
                    continue
                if (u == x and v == y) or (u == y and v == x):
                    continue
                swap_valid = True
                break
            
            if not swap_valid:
                continue
                
            # Apply swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            new_max, new_sum_sq = compute_books(adj)
            
            # Check improvement
            if new_max < cur_max or (new_max == cur_max and new_sum_sq < cur_sum_sq):
                cur_max, cur_sum_sq = new_max, new_sum_sq
                stuck_count = 0
                if cur_max < best_max or (cur_max == best_max and cur_sum_sq < best_sum_sq):
                    best_max, best_sum_sq = cur_max, cur_sum_sq
                    best_adj = adj.copy()
                    if best_max == 7:
                        return best_adj
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                stuck_count += 1
                
            # Kick mechanism
            if stuck_count >= kick_threshold:
                # Perform random perturbations
                for _ in range(kick_size):
                    if random_swap(adj, rng):
                        # We don't update cur_max/cur_sum_sq here, we just move the state
                        # The next iteration will evaluate it.
                        # But we need to update cur_max/cur_sum_sq to reflect the new state for future comparisons?
                        # No, the greedy loop compares against cur_max. If we kick, the state changes.
                        # We should recompute cur_max/cur_sum_sq after a kick to reset the baseline.
                        pass
                # Recompute baseline after kick
                cur_max, cur_sum_sq = compute_books(adj)
                stuck_count = 0
                
        return best_adj

    # Initialization: K_20,20 + 1 edge
    # This is a bipartite graph with parts A={0..19} and B={20..39}.
    # Edges are all (a,b) for a in A, b in B.
    # This has 20*20 = 400 edges.
    # We add 1 edge within A or within B. Adding within A creates a triangle?
    # If we add (0,1) in A, triangles involving (0,1) need a common neighbor in B.
    # 0 and 1 are connected to all 20 vertices in B. So book size of (0,1) is 20.
    # This is bad.
    # Better initialization: Start with K_20,20.
    # Add an edge that minimizes immediate book increase?
    # Any edge within a part creates a book of size 20.
    # So max_book starts at 20.
    # We want to reduce this.
    # The local search will try to swap edges to break these large books.
    
    adj = [0] * n
    for i in range(20):
        for j in range(20, 40):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Add 1 random edge within one of the parts to reach 401 edges
    # Let's pick a random pair in A
    u, v = 0, 1
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    best_adj = adj
    best_max, _ = compute_books(adj)
    
    rng = random.Random(42)
    
    # Run greedy search
    # Allocate most of the time to this
    search_deadline = start_time + 140
    result_adj = run_greedy_search(adj, search_deadline, rng)
    
    res_max, _ = compute_books(result_adj)
    if res_max < best_max:
        best_adj = result_adj
        best_max = res_max
        
    # If we have time, do a few more restarts with different kicks?
    # Or just return the best found.
    # Given the strict greedy nature, it might get stuck.
    # Let's do a few quick restarts if time permits.
    
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Restart from best_adj with a random kick
            trial_adj = best_adj.copy()
            for _ in range(10):
                random_swap(trial_adj, rng)
            trial_adj = run_greedy_search(trial_adj, deadline - 5, rng)
            t_max, _ = compute_books(trial_adj)
            if t_max < best_max:
                best_adj = trial_adj
                best_max = t_max

    return get_edge_list(best_adj)

