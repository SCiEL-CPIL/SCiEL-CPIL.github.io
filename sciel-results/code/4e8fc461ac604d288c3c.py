"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_index = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx

def _solve_2sat(n_vars, clauses):
    """Solve 2-SAT. n_vars: number of variables (0..n_vars-1).
    Each clause is a pair (a, b) meaning (a OR b) where a,b are signed ints.
    Variable i is positive if i > 0, negative if i < 0. Var 0 means no constraint.
    Returns assignment list of booleans (True = positive) or None if unsat.
    """
    if not clauses:
        return [False] * n_vars
    
    # Build implication graph
    # Node encoding: var i -> 2*i (False), 2*i+1 (True)
    # Clause (a, b): (!a -> b) and (!b -> a)
    # !a: if a>0 then -a, if a<0 then -a (flip sign)
    # Actually: node for literal L: if L>0, node = 2*(L-1)+1; if L<0, node = 2*(-L-1)
    # Negation of literal L: if L>0, neg = -(L); if L<0, neg = -L
    
    def lit_to_node(lit):
        if lit > 0:
            return 2 * (lit - 1) + 1  # True
        else:
            return 2 * (-lit - 1)      # False
    
    def negate_lit(lit):
        return -lit
    
    total_nodes = 2 * n_vars
    adj = [[] for _ in range(total_nodes)]
    
    for (a, b) in clauses:
        # (!a -> b) and (!b -> a)
        na = negate_lit(a)
        nb = negate_lit(b)
        adj[lit_to_node(na)].append(lit_to_node(b))
        adj[lit_to_node(nb)].append(lit_to_node(a))
    
    # Kosaraju's algorithm for SCC
    visited = [False] * total_nodes
    order = []
    
    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)
    
    for i in range(total_nodes):
        if not visited[i]:
            dfs1(i)
    
    radj = [[] for _ in range(total_nodes)]
    for u in range(total_nodes):
        for v in adj[u]:
            radj[v].append(u)
    
    comp = [-1] * total_nodes
    comp_id = 0
    
    def dfs2(u, cid):
        comp[u] = cid
        for v in radj[u]:
            if comp[v] == -1:
                dfs2(v, cid)
    
    for u in reversed(order):
        if comp[u] == -1:
            dfs2(u, comp_id)
            comp_id += 1
    
    # Check for contradiction
    for i in range(n_vars):
        if comp[2 * i] == comp[2 * i + 1]:
            return None
    
    # Assign: var i is True if comp[2*i+1] > comp[2*i] (higher comp id = processed later in topological order)
    assignment = [False] * n_vars
    for i in range(n_vars):
        assignment[i] = comp[2 * i + 1] > comp[2 * i]
    
    return assignment

def _hamilton_cycle_2sat(K):
    """Find Hamilton cycle using edges with index <= K via 2-SAT.
    Returns a permutation (cycle) or None.
    """
    # Build adjacency: adj[u] = set of v such that edge (u,v) has index <= K
    adj = [set() for _ in range(N)]
    for (u, v), idx in edge_index.items():
        if idx <= K:
            adj[u].add(v)
            adj[v].add(u)
    
    # Pre-check: every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # 2-SAT formulation for Hamilton cycle:
    # Fix vertex 0 as start.
    # For each vertex v != 0, we define a variable x_v.
    # x_v = True means: in the cycle, v's predecessor is "before" v in some ordering... 
    # 
    # Better formulation: Use the standard 2-SAT for Hamilton cycle.
    # We have n-1 variables (for vertices 1..n-1).
    # Variable x_i for vertex i (i >= 1): 
    #   x_i = True means vertex i is reached from a vertex with smaller index in the path from 0
    #   x_i = False means vertex i is reached from a vertex with larger index
    #
    # Actually, let me use a cleaner formulation:
    # We want to find a permutation p_0=0, p_1, ..., p_{n-1} such that:
    #   (p_{n-1}, p_0) is an edge, and (p_i, p_{i+1}) are edges for all i.
    #
    # 2-SAT approach: For each pair (i,j) with i<j, i,j != 0, define variable y_{i,j}.
    # y_{i,j} = True means i comes before j in the cycle (i appears earlier in the path from 0).
    # y_{i,j} = False means j comes before i.
    #
    # This has C(n-1,2) variables which is C(23,2) = 253. That's a lot but manageable.
    #
    # Constraints:
    # 1. Transitivity: y_{i,j} AND y_{j,k} => y_{i,k}, i.e., (!y_{i,j} OR !y_{j,k} OR y_{i,k})
    #    But 2-SAT only handles 2 literals per clause. So we need to break this up.
    #    Actually, transitivity gives 3-SAT constraints, not 2-SAT.
    #
    # Let me use a different approach. The standard 2-SAT for Hamilton cycle uses:
    # For each vertex v (except start s), and each pair of neighbors (a,b) of v,
    # we have a constraint that v cannot be "between" a and b if that creates a subcycle.
    #
    # Actually, the cleanest 2-SAT formulation:
    # Variables: For each vertex v != s, variable x_v.
    # x_v = True means v's predecessor in the cycle has index < v (in vertex numbering)
    # x_v = False means v's predecessor has index > v
    #
    # Hmm, this doesn't directly work either.
    #
    # Let me use a well-known 2-SAT formulation:
    # We fix the start vertex s = 0.
    # For each vertex v != s, we have two "ports": incoming and outgoing.
    # The key insight: in a Hamilton cycle, each vertex has exactly one predecessor and one successor.
    #
    # Alternative: Use the fact that a Hamilton cycle is equivalent to a 2-regular connected spanning subgraph.
    # The 2-regular part is easy (each vertex degree 2), but connectivity is hard.
    #
    # For small n, let me just use a smarter DFS with strong pruning.
    # But the current DFS is too slow.
    #
    # Let me try a different strategy: use the edge list directly and do a greedy + repair approach,
    # or use a bitmask DP which is O(n^2 * 2^n) = too slow for n=24.
    #
    # Actually, for n=24, 2^24 = 16M states, times 24 = ~400M operations. Too slow in Python.
    #
    # Let me optimize the DFS significantly with better pruning and ordering.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    max_idx = max(edge_index.values())
    
    # Binary search for smallest K with a Hamilton cycle
    # Use optimized DFS
    
    def exists_hamiltonian_cycle(K):
        adj = [[] for _ in range(N)]
        allowed = [[False] * N for _ in range(N)]
        
        for (u, v), idx in edge_index.items():
            if idx <= K:
                adj[u].append(v)
                adj[v].append(u)
                allowed[u][v] = True
                allowed[v][u] = True
        
        # Sort neighbors by degree (Warnsdorff heuristic)
        for i in range(N):
            adj[i].sort(key=lambda x: len(adj[x]))
        
        # Pre-check
        for i in range(N):
            if len(adj[i]) < 2:
                return None
        
        start = 0
        full_mask = (1 << N) - 1
        
        # Use iterative DFS with explicit stack to avoid recursion limits
        # and to allow time checking
        stack = [(start, 1 << start, [start])]
        
        while stack:
            if time.time() > deadline:
                return None
            
            u, mask, path = stack[-1]
            
            if mask == full_mask:
                if allowed[u][start]:
                    return path[:]
                stack.pop()
                continue
            
            # Find next unvisited neighbor
            found = False
            for v in adj[u]:
                if not (mask & (1 << v)):
                    # Pruning: check if v can lead to completion
                    # Simple check: v must have at least one unvisited neighbor (other than u)
                    # or be the last vertex (connecting to start)
                    new_mask = mask | (1 << v)
                    if new_mask == full_mask:
                        if allowed[v][start]:
                            return path + [v]
                    else:
                        # Check that v has at least one unvisited neighbor
                        has_next = False
                        for w in adj[v]:
                            if not (new_mask & (1 << w)):
                                has_next = True
                                break
                        if not has_next:
                            continue
                        # Check connectivity of unvisited vertices
                        # The unvisited vertices (excluding v) must be connected
                        # in the induced subgraph, plus v must connect to them
                        unvisited = [x for x in range(N) if not (new_mask & (1 << x))]
                        if unvisited:
                            # BFS from v through unvisited vertices
                            visited_set = set()
                            queue = [v]
                            visited_set.add(v)
                            while queue:
                                cur = queue.pop(0)
                                for w in adj[cur]:
                                    if (new_mask & (1 << w)) == 0 and w not in visited_set:
                                        visited_set.add(w)
                                        queue.append(w)
                            if len(visited_set) - 1 != len(unvisited):
                                # Not all unvisited vertices reachable from v
                                continue
                    stack.append((v, new_mask, path + [v]))
                    found = True
                    break
            
            if not found:
                stack.pop()
        
        return None
    
    low = 0
    high = max_idx
    best_cycle = None
    
    while low <= high:
        mid = (low + high) // 2
        cycle = exists_hamiltonian_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
    
    if best_cycle is None:
        return list(range(N))
    
    # Local search: try to reduce max edge index by swapping edges
    def cycle_score(cycle):
        max_e = 0
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            a, b = (u, v) if u < v else (v, u)
            idx = edge_index[(a, b)]
            if idx > max_e:
                max_e = idx
        return max_e
    
    current_score = cycle_score(best_cycle)
    
    # Try 2-opt style swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        cycle = best_cycle[:]
        for i in range(N):
            for j in range(i + 1, N):
                # Reverse segment i..j
                new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < current_score:
                    best_cycle = new_cycle
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return best_cycle

# EVOLVE-BLOCK-END
