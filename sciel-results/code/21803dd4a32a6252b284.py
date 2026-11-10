"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    N = 100
    target = 200

    def compute_reachable_sums(elements):
        """Compute set of all subset sums up to target using numpy for speed."""
        reachable = np.zeros(target + 1, dtype=bool)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            reachable[x:] |= reachable[:-x]
        return reachable

    def can_add_element(x, current_reachable):
        """Check if adding x to current_set keeps it free of subset sum 200."""
        if x > target:
            return False
        if 200 - x <= target and current_reachable[200 - x]:
            return False
        return True

    def max_flow_min_cut_flow_network():
        """
        Build a flow network to find minimum elements to remove so no subset sums to 200.
        
        Network structure:
        - Source node
        - Element nodes (one per x in 1..100), capacity 1 from source
        - Sum-state nodes (one per sum 0..199)
        - Sink node
        
        Edges:
        - source -> element_x (cap 1)
        - element_x -> sum_state_{s+x} for all s where s+x <= 199 (cap INF)
        - sum_state_s -> sink if s == 200... but 200 is not a valid intermediate state.
        
        Actually, let's think of it as: we want to prevent any path from source to a "target" 
        that represents reaching sum 200. 
        
        Alternative: Use the dual. The maximum subset with no subset sum 200 equals 
        N - (minimum number of elements to remove to break all subset-sum-200 paths).
        
        Let me use a different approach: iterative removal guided by flow.
        """
        pass

    def greedy_largest_first():
        selected = []
        reachable = {0}
        for x in range(N, 0, -1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable

    def greedy_smallest_first():
        selected = []
        reachable = {0}
        for x in range(1, N + 1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable

    def find_subset_sum_fast(elements):
        """Find a subset of elements summing to target, return it or None."""
        n = len(elements)
        dp = [-1] * (target + 1)
        dp[0] = -1
        parent = {}

        for i, x in enumerate(elements):
            for s in range(target - x, -1, -1):
                if dp[s] != -1 and dp[s + x] == -1:
                    dp[s + x] = i
                    parent[(s + x, i)] = s
            if dp[target] != -1:
                subset = []
                cur = target
                while cur > 0:
                    i = dp[cur]
                    subset.append(elements[i])
                    cur = parent[(cur, i)]
                return subset
        return None

    def iterative_remove_optimal(removal_strategy='critical', max_iter=500):
        """Iteratively remove elements to break all subset-sum-200 paths."""
        remaining = list(range(1, N + 1))
        removed = set()

        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break

            subset = find_subset_sum_fast(remaining)
            if subset is None:
                break

            if removal_strategy == 'largest':
                to_remove = max(subset)
            elif removal_strategy == 'smallest':
                to_remove = min(subset)
            elif removal_strategy == 'random':
                to_remove = random.choice(subset)
            elif removal_strategy == 'critical':
                sorted_subset = sorted(subset)
                mid = len(sorted_subset) // 2
                to_remove = sorted_subset[mid]
            else:
                to_remove = max(subset)

            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]

        return remaining

    # === NEW STRUCTURAL APPROACH: Max-flow inspired min-cut ===
    def flow_guided_removal():
        """
        Use max-flow concepts to guide removal.
        
        Key idea: Build a bipartite graph between "sum states" and "elements".
        For each sum state s (0..199) and element x, if s+x <= 199, there's a 
        transition. We want to find the minimum set of elements to remove so that
        no path exists from sum 0 to sum 200.
        
        This is a minimum vertex cut problem in a DAG. We solve it via max-flow.
        """
        # Build the DAG of sum states
        # Nodes: (sum_state, element_index) is too complex. 
        # Instead: layered graph where layer i = sum after i elements.
        # But number of elements varies. 
        #
        # Simpler: For each sum s in 0..199, and each element x, 
        # edge from s to s+x (if s+x <= 199). The "cost" of using element x 
        # is 1 (we pay 1 to include it). We want to find the minimum cost 
        # set of elements whose removal disconnects 0 from 200.
        #
        # This is a minimum s-t cut in a graph where each element x has 
        # capacity 1, and sum-state transitions have infinite capacity.
        #
        # Graph:
        # - Source = 0
        # - Sink = 200
        # - For each sum s in 0..199, node s
        # - For each element x in 1..100, node x (element node)
        # - Edges: s -> x (if s+x <= 199, cap INF), x -> s+x (cap 1)
        #   Wait, that doesn't work. Let me rethink.
        #
        # Actually: To use element x to go from sum s to sum s+x, we must 
        # "pay" for x. So: edge from sum_node_s to elem_node_x (cap INF), 
        # then elem_node_x to sum_node_{s+x} (cap 1). But x can be used in 
        # multiple transitions, so the capacity 1 on elem_node_x should be 
        # shared. This requires splitting the element node.
        #
        # Standard trick: split each element node x into x_in and x_out 
        # with capacity 1. Then: sum_s -> x_in (INF), x_in -> x_out (1), 
        # x_out -> sum_{s+x} (INF) for all valid s.
        #
        # Source = sum_0, Sink = sum_200.
        # Min cut = minimum elements to remove.
        
        INF = 10**9
        
        # Build adjacency list for max-flow
        # Nodes: 
        #   sum_node(s) = s (0..199)
        #   sum_node(200) = 200 (sink)
        #   elem_in(x) = 1000 + x
        #   elem_out(x) = 2000 + x
        #   source = 0 (sum 0)
        #   sink = 200
        
        num_nodes = 3000
        # Adjacency: adj[u] = list of (v, capacity)
        adj = [[] for _ in range(num_nodes)]
        # Reverse edges for residual graph
        rev = [None] * num_nodes
        # Edge capacities
        cap = [0] * num_nodes
        
        def add_edge(u, v, c):
            adj[u].append([v, c, len(adj[v])])
            adj[v].append([u, 0, len(adj[u]) - 1])
        
        # Add element split edges
        for x in range(1, N + 1):
            elem_in = 1000 + x
            elem_out = 2000 + x
            add_edge(elem_in, elem_out, 1)
        
        # Add sum-state to element and element to sum-state edges
        for s in range(0, 200):  # sum states 0..199
            for x in range(1, N + 1):
                if s + x <= 199:
                    elem_in = 1000 + x
                    # sum_s -> elem_in (INF)
                    add_edge(s, elem_in, INF)
                    # elem_out -> sum_{s+x} (INF)
                    elem_out = 2000 + x
                    add_edge(elem_out, s + x, INF)
        
        # Source = 0, Sink = 200
        source = 0
        sink = 200
        
        # Run max-flow (Edmonds-Karp or Dinic)
        # Use Dinic for speed
        def dinic_max_flow():
            flow = 0
            n = num_nodes
            
            while time.time() < deadline - 5:
                # BFS to find level
                level = [-1] * n
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for v, c, rev_idx in adj[u]:
                        if c > 0 and level[v] < 0:
                            level[v] = level[u] + 1
                            q.append(v)
                
                if level[sink] < 0:
                    break
                
                # DFS to find blocking flow
                it = [0] * n
                
                def dfs(u, f):
                    if u == sink:
                        return f
                    for i in range(it[u], len(adj[u])):
                        it[u] = i
                        v, c, rev_idx = adj[u][i]
                        if c > 0 and level[v] == level[u] + 1:
                            d = dfs(v, min(f, c))
                            if d > 0:
                                adj[u][i][1] -= d
                                adj[v][rev_idx][1] += d
                                return d
                    return 0
                
                while True:
                    if time.time() > deadline - 5:
                        return flow
                    f = dfs(source, INF)
                    if f == 0:
                        break
                    flow += f
            
            return flow
        
        # Actually, for this problem, we don't need the full max-flow value.
        # We need the min-cut set. Let me just run the flow and then 
        # find the min cut.
        
        # For now, use the flow to guide iterative removal.
        # If the min-cut is small, we can find it directly.
        
        # Let me just use the iterative approach with the flow as a guide.
        # The flow value tells us the minimum number of removals needed.
        
        # For simplicity, just return the greedy result for now and 
        # use flow later if time permits.
        return None

    # Greedy from large to small:
    selected, reachable = greedy_largest_first()
    best_selected = list(selected)
    best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Try greedy from small to large:
    sel2, reach2 = greedy_smallest_first()
    if len(sel2) > best_size:
        best_selected = list(sel2)
        best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Try different removal strategies
    for strategy in ['largest', 'smallest', 'critical']:
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal(strategy, max_iter=300)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Random restarts with random removal order
    random.seed(42)

    for trial in range(200):
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal('random', max_iter=200)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Local search: try to add back removed elements
    all_elements = set(range(1, N + 1))
    current_set = set(best_selected)

    # Compute reachable sums for current_set
    current_reachable = compute_reachable_sums(list(current_set))

    for _ in range(100):  # iterations of local search
        if time.time() > deadline - 2:
            break
        improved = False
        candidates = list(all_elements - current_set)
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline - 2:
                break
            if can_add_element(x, current_reachable):
                current_set.add(x)
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break

        if not improved:
            break

    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)

    # Second pass local search with different ordering
    current_set = set(best_selected)
    current_reachable = compute_reachable_sums(list(current_set))

    for _ in range(100):
        if time.time() > deadline - 1:
            break
        improved = False
        candidates = sorted(all_elements - current_set, reverse=True)

        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add_element(x, current_reachable):
                current_set.add(x)
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break

        if not improved:
            break

    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)

    return best_selected
# EVOLVE-BLOCK-END
