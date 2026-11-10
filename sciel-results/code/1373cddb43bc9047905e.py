import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.count = n  # number of components
            
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
            
        def union(self, x, y):
            px = self.find(x)
            py = self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            self.count -= 1
            return True

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for I vertices to C vertices
        # adj[i] is a set of C vertices
        adj_I = [set() for _ in range(n)]
        C_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                C_edges.add((u, v))
            elif u >= k and v < k:
                adj_I[u].add(v)
                adj_I[v].add(u)
            elif u < k and v >= k:
                adj_I[v].add(u)
                adj_I[u].add(v)
            else:
                # I-I edge, should not exist in split graph definition given
                pass

        I_vertices = list(range(k, n))
        C_vertices = list(range(k))
        
        # Initialize DSU for each I vertex
        # We only care about vertices in N(i)
        # Map C vertex to local index in N(i) for DSU? 
        # Or just use global C vertex indices and a DSU that only unions those present.
        # Simpler: DSU over 0..k-1, but only consider N(i).
        # We need to know which vertices are in N(i) to count components.
        
        dsus = {}
        N_i = {}
        for i in I_vertices:
            N_i[i] = adj_I[i]
            if N_i[i]:
                dsus[i] = DSU(k)
            else:
                dsus[i] = None

        # List of all C edges
        all_C_edges = list(C_edges)
        
        # Assignment: edge -> i
        # We want to assign each edge to an i to minimize sum of components.
        # Greedy: For each edge, pick i that reduces component count the most.
        # If tie, pick i with smallest current component count? Or largest?
        # Reducing component count is the main goal.
        
        # To make it faster, we can sort edges by "popularity" (number of i's containing both ends)
        # Edges with many choices are more flexible.
        
        edge_choices = {}
        for e in all_C_edges:
            u, v = e
            choices = []
            for i in I_vertices:
                if u in N_i[i] and v in N_i[i]:
                    choices.append(i)
            edge_choices[e] = choices
            
        # Sort edges by number of choices (ascending? descending?)
        # If an edge has few choices, it's more constrained. Assign it first?
        # If we assign it to the only choice, we might merge components.
        # If we leave it unassigned, it goes to pure cover.
        # Let's try assigning all edges that CAN be assigned.
        
        # We need to track which edges are assigned.
        assigned_edges = set()
        
        # Process edges
        # To improve, we can iterate multiple times or use a priority queue.
        # But simple greedy should be good.
        
        # Sort by length of choices
        sorted_edges = sorted(all_C_edges, key=lambda e: len(edge_choices[e]))
        
        for e in sorted_edges:
            u, v = e
            choices = edge_choices[e]
            if not choices:
                continue
                
            best_i = None
            best_gain = 0
            
            for i in choices:
                dsu = dsus[i]
                if dsu is None: continue
                # Check if union would reduce count
                pu = dsu.find(u)
                pv = dsu.find(v)
                if pu != pv:
                    gain = 1
                else:
                    gain = 0
                
                # Prefer higher gain
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                elif gain == best_gain and best_i is not None:
                    # Tie breaker: pick i with fewer components? 
                    # Or just first one.
                    pass
            
            if best_i is not None and best_gain > 0:
                # Assign e to best_i
                dsus[best_i].union(u, v)
                assigned_edges.add(e)
            elif best_i is not None:
                # Gain is 0, but we can still assign it to avoid pure cover cost?
                # Assigning it doesn't change I-clique count.
                # But it removes it from pure cover.
                # Pure cover cost is at least 1 if any edge remains.
                # So assigning is always >= as good as not assigning?
                # Yes, because pure cover cost is convex?
                # Actually, if we assign it, we don't pay for it in pure cover.
                # If we don't, we might pay for it.
                # So assign it to ANY choice.
                dsus[best_i].union(u, v) # No effect on count
                assigned_edges.add(e)

        # Now, calculate I-cliques
        result_cliques = []
        
        for i in I_vertices:
            if dsus[i] is None:
                continue
                
            # Get components
            components = defaultdict(list)
            for c in N_i[i]:
                root = dsus[i].find(c)
                components[root].append(c)
                
            for root, members in components.items():
                if len(members) >= 1:
                    # Clique is {i} U members
                    # Must have at least 2 vertices.
                    # If len(members) == 0, skip (shouldn't happen if N_i not empty)
                    # If len(members) >= 1, clique size is 1 + len(members) >= 2.
                    clique = [i] + members
                    result_cliques.append(clique)
                    
        # Remaining C edges
        remaining_C_edges = C_edges - assigned_edges
        
        if remaining_C_edges:
            # Cover with pure C cliques
            # Build graph of remaining edges
            v_set = set()
            for u, v in remaining_C_edges:
                v_set.add(u)
                v_set.add(v)
                
            adj_sub = {v: set() for v in v_set}
            for u, v in remaining_C_edges:
                adj_sub[u].add(v)
                adj_sub[v].add(u)
                
            # Greedy Max Clique
            remaining = set(remaining_C_edges)
            while remaining:
                # Find max clique
                # Simple greedy: pick vertex with max degree, expand
                if not v_set: break
                candidates = list(v_set)
                candidates.sort(key=lambda x: len(adj_sub[x]), reverse=True)
                
                # Use a simple recursive search for max clique for small graphs
                # Or just greedy
                best_clique = []
                # Try to build a clique starting from each candidate
                # For speed, just pick the first candidate and expand
                # This is not optimal but fast.
                
                # Better: Bron-Kerbosch with pivot? 
                # Given time, let's use a simple heuristic:
                # Pick vertex with max degree in remaining graph
                # Add it to clique
                # Remove it and non-neighbors
                # Repeat
                
                curr_v_set = set(v_set)
                curr_adj = {v: set(adj_sub[v] & curr_v_set) for v in curr_v_set}
                
                clique = []
                while curr_v_set:
                    # Pick max degree
                    v = max(curr_v_set, key=lambda x: len(curr_adj[x]))
                    clique.append(v)
                    # Update
                    neighbors = curr_adj[v]
                    curr_v_set.remove(v)
                    new_curr_v_set = set()
                    for u in curr_v_set:
                        if u in neighbors:
                            new_curr_v_set.add(u)
                    curr_v_set = new_curr_v_set
                    # Update adj for remaining
                    curr_adj = {u: set(curr_adj[u] & curr_v_set) for u in curr_v_set}
                    
                if clique:
                    result_cliques.append(clique)
                    # Remove edges covered by this clique
                    for u in clique:
                        for v in clique:
                            if u < v:
                                remaining.discard((u, v))
                            else:
                                remaining.discard((v, u))
                else:
                    # Should not happen if remaining is not empty
                    break

        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
