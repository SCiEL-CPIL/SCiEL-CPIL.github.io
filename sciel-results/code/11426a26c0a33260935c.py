import time
import numpy as np
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        """
        Checks if the graph has girth at least g.
        Returns the girth if < g, else g (or larger).
        Uses non-backtracking BFS from every vertex.
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        if d + 1 < g:
                            queue.append(nb)
        
        if min_cycle < g:
            return min_cycle
        return min_cycle if min_cycle != float('inf') else g

    # ---------------- Direct Backtracking Construction ----------------
    def construct_by_backtracking():
        """
        Build a 3-regular graph of girth >= g by growing a non-backtracking tree
        from a root, then merging boundary vertices to close the graph.
        
        Strategy:
        1. Grow a non-backtracking tree of depth d = g//2 from root vertex 0.
        2. The tree has boundary vertices at depth d.
        3. Merge boundary vertices in pairs (or groups) to satisfy 3-regularity.
        4. Verify no short cycles are created.
        """
        if k != 3 or g != 16:
            return None
            
        depth = g // 2  # 8 for g=16
        
        # Grow non-backtracking tree from root
        # Each node in the tree is identified by a path (sequence of directions)
        # We'll represent the tree explicitly
        
        # Tree nodes: (id, parent_id, depth, children)
        # We use a list-based representation
        
        tree_nodes = []  # list of dicts: {'id': int, 'parent': int, 'depth': int, 'children': list}
        node_id = 0
        
        # Root
        tree_nodes.append({'id': 0, 'parent': -1, 'depth': 0, 'children': []})
        
        # Grow tree level by level
        # For a 3-regular graph, each node has 3 neighbors.
        # In the non-backtracking tree, each node (except root) has 2 new children.
        # Root has 3 children.
        
        for d in range(depth):
            if time.time() > deadline - 30:
                return None
                
            # Get all nodes at current depth
            current_level = [n for n in tree_nodes if n['depth'] == d]
            
            for node in current_level:
                # Determine how many children this node should have
                if node['depth'] == 0:
                    num_children = 3  # root has 3 children
                else:
                    num_children = 2  # other nodes have 2 new children (one is parent)
                
                for c in range(num_children):
                    child_id = len(tree_nodes)
                    tree_nodes.append({'id': child_id, 'parent': node['id'], 
                                       'depth': d + 1, 'children': []})
                    node['children'].append(child_id)
        
        n_tree = len(tree_nodes)
        
        # Get boundary vertices (at maximum depth)
        boundary = [n['id'] for n in tree_nodes if n['depth'] == depth]
        n_boundary = len(boundary)
        
        # For g=16, depth=8:
        # Level 0: 1 node
        # Level 1: 3 nodes
        # Level 2: 6 nodes
        # Level 3: 12 nodes
        # Level 4: 24 nodes
        # Level 5: 48 nodes
        # Level 6: 96 nodes
        # Level 7: 192 nodes
        # Level 8: 384 nodes
        # Total: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 766 nodes
        # Boundary: 384 nodes
        
        # Each boundary node needs 1 more edge to reach degree 3.
        # We need to pair up boundary nodes with additional edges.
        # But we must avoid creating cycles shorter than g.
        
        # Strategy: Randomly permute boundary nodes and pair them up.
        # Then verify the girth.
        
        # To make this more efficient, we'll try multiple random pairings.
        
        # First, let's understand the structure better.
        # Each boundary node at depth 8 has exactly one "free" edge slot.
        # We need to add n_boundary/2 edges to pair them up.
        
        # But wait - this naive approach creates a graph where all cycles go through
        # the root, which might create short cycles.
        
        # Better approach: Use a more sophisticated construction.
        # Let's try a different strategy: build the graph incrementally.
        
        # Alternative: Use the fact that we need a 3-regular graph.
        # Start with a random 3-regular graph on n vertices and check girth.
        # This is unlikely to work for large n.
        
        # Let's try a hybrid: use the tree structure but merge more cleverly.
        
        # Actually, let me reconsider. The voltage lift approach is more principled.
        # Let me improve it instead.
        
        return None

    # ---------------- Improved Voltage Lift ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all non-backtracking closed walks of length [5, 15]
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 5:
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
                    
        if not walk_vectors:
            return None
            
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None

        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        C_cols = [C[:, j] for j in range(15)]
        
        def solve_voltage(m, max_iters=50000):
            """
            Uses Simulated Annealing to find a voltage assignment v
            such that no short cycle exists.
            """
            # Initialize with random values
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)

            best_v = v.copy()
            best_cost = cost(v)

            if best_cost == 0:
                return v, 0

            T = 5.0
            cooling = 0.9998
            min_T = 0.01
            stall_count = 0
            max_stall = 1000
            
            for it in range(max_iters):
                if remaining() < 10:
                    break

                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        stall_count = 0
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                    stall_count += 1

                T *= cooling

                if stall_count > max_stall:
                    v = np.random.randint(1, m, size=15).astype(np.int64)
                    new_cost = cost(v)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    T = 3.0
                    stall_count = 0

            return best_v, best_cost

        # Try primes in increasing order
        m = 53
        best_result = None
        
        while m < 500 and remaining() > 30:
            if is_prime(m):
                time_left = remaining()
                num_restarts = max(5, min(20, int(time_left / 5)))
                for restart in range(num_restarts):
                    if remaining() < 15:
                        break

                    v, cost = solve_voltage(m, max_iters=5000)

                    if cost == 0:
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                        
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc >= g:
                            return (n, edges)
                        
            m += 1
            
        return None

    # Try voltage lift first (most likely to succeed)
    res = voltage_lift_petersen()
    if res is not None:
        return res
        
    # Fallback: try backtracking
    res = construct_by_backtracking()
    if res is not None:
        return res
        
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
