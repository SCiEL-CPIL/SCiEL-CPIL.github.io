import time
import random
from collections import defaultdict, deque
from itertools import combinations

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph_ilp(edges, n, time_limit=10):
    """
    Decomposes the graph into cycles and single edges.
    Strategy:
    1. Use ILP to find a 2-factor (max number of edges in cycles).
    2. Extract cycles from the 2-factor.
    3. Use 2-opt to merge cycles into larger cycles (reducing piece count).
    4. Remaining edges are single edges.
    """
    if not edges:
        return []
    
    # Map edges to indices
    edge_list = list(edges)
    m = len(edge_list)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    
    # Adjacency list
    adj = defaultdict(set)
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        
    # Degrees
    deg = [len(adj[u]) for u in range(n)]
    
    # Variables: x_i for each edge (0 or 1)
    # Objective: Maximize sum(x_i)
    # Constraints:
    # 1. For each vertex u: sum_{e incident to u} x_e <= 2
    # 2. For each subset S of vertices (|S| < n): sum_{e in E(S)} x_e <= |S| - 1
    #    (To prevent subtours. This is exponential, so we might need a heuristic or just rely on the fact
    #     that in a sparse graph, subtours are less likely to be optimal if we maximize edges? 
    #     Actually, maximizing edges in a 2-factor IS the 2-factor problem.
    #     Without subtour elimination, we might get a 2-matching with cycles of length 2? 
    #     No, graph is simple, so min cycle length is 3.
    #     A 2-matching can have multiple cycles. That is exactly what we want (a 2-factor is a 2-matching 
    #     that covers all vertices? No, 2-factor covers all vertices. A 2-matching covers a subset).
    #     We want to maximize the number of edges in a subgraph where deg(v) <= 2.
    #     This subgraph will be a collection of cycles and paths.
    #     Wait, if deg(v) <= 2, components are paths or cycles.
    #     We want to maximize edges. This will prefer cycles over paths if possible?
    #     Actually, a path of length L has L edges and L+1 vertices. A cycle of length L has L edges and L vertices.
    #     So cycles are "more efficient" in terms of edges per vertex.
    #     The ILP will find the maximum 2-matching.
    #     Then we extract cycles from it. The remaining edges in the 2-matching are paths.
    #     We can treat path edges as single edges or try to extend them?
    #     For simplicity, we extract cycles from the 2-matching. The rest of the 2-matching edges 
    #     (which form paths) will be added as single edges. The edges NOT in the 2-matching 
    #     will also be added as single edges.
    #     Wait, this is suboptimal. If we have a path u-v-w, we could just make it two single edges [u,v], [v,w].
    #     Or if we can connect u to w, we make a cycle. But u-w might not be an edge.
    #     So, extracting cycles from the 2-matching is a good first step.
    
    if not HAS_SCIPY:
        # Fallback to greedy if no scipy
        return greedy_decompose(edges, n)
        
    # Build A_ub for degree constraints
    # A is n x m
    rows = []
    cols = []
    vals = []
    for i, (u, v) in enumerate(edge_list):
        rows.append(u)
        cols.append(i)
        vals.append(1)
        rows.append(v)
        cols.append(i)
        vals.append(1)
        
    A_deg = csr_matrix((vals, (rows, cols)), shape=(n, m))
    
    # Objective: -1 for all edges (minimize -sum)
    c = -1.0 * ones(m) # Need numpy
    
    # Bounds
    lb = zeros(m)
    ub = ones(m)
    
    # Constraints
    # Degree <= 2
    cons_deg = LinearConstraint(A_deg, lb=0, ub=2)
    
    # Subtour elimination is hard. Let's try without first.
    # If the result has subtours (cycles), that's fine.
    # If it has paths, we'll handle them later.
    
    res = milp(c=c, constraints=[cons_deg], integrality=ones(m), bounds=Bounds(lb, ub))
    
    if not res.success:
        return greedy_decompose(edges, n)
        
    x = res.x
    selected_edges = [edge_list[i] for i in range(m) if x[i] > 0.5]
    selected_set = set(tuple(sorted(e)) for e in selected_edges)
    
    # Extract cycles from selected edges
    adj_sel = defaultdict(set)
    for u, v in selected_edges:
        adj_sel[u].add(v)
        adj_sel[v].add(u)
        
    visited = set()
    cycles = []
    
    for u in range(n):
        if u in visited:
            continue
        if u not in adj_sel or len(adj_sel[u]) != 2:
            # Not part of a cycle (could be path endpoint or isolated)
            # But if deg is 1, it's a path. If deg is 0, isolated.
            # If deg is 2, it's a cycle.
            continue
            
        # Find cycle
        path = [u]
        visited.add(u)
        prev = -1
        curr = u
        while True:
            nexts = list(adj_sel[curr])
            # Pick the one that isn't prev
            nxt = nexts[0] if nexts[0] != prev else nexts[1]
            if nxt == u:
                break
            if nxt in visited:
                # Should not happen in a simple cycle unless we closed it
                break
            path.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
            
        if len(path) >= 3:
            cycles.append(path)
            
    # Now we have a set of vertex-disjoint cycles.
    # The remaining edges in the graph are:
    # 1. Edges in the 2-matching that are not in cycles (paths)
    # 2. Edges not in the 2-matching
    
    # We want to minimize total pieces.
    # Current pieces: cycles + (all other edges as single edges)
    
    # Can we merge cycles?
    # 2-opt: For two cycles C1 and C2, if there are edges (u1, v1) in C1 and (u2, v2) in C2
    # such that (u1, u2) and (v1, v2) are edges in the original graph,
    # we can swap them to merge C1 and C2 into one larger cycle.
    
    # Let's implement a simple local search.
    
    # Represent cycles as lists of vertices
    # We need to know which edges are in which cycle
    # cycle_edges: list of sets of edges
    
    # Actually, it's easier to work with the adjacency of the current "solution"
    # which consists of cycles and single edges.
    
    # Let's build a structure:
    # pieces: list of cycles (lists of vertices)
    # single_edges: set of edges
    
    # Initially:
    # pieces = cycles found above
    # single_edges = all edges not in any of those cycles
    
    # Wait, the 2-matching might have paths.
    # E.g. u-v-w. u-v and v-w are in selected_edges.
    # These are not in any cycle.
    # So they go to single_edges.
    
    # Let's identify which edges are in the cycles found.
    cycle_edge_set = set()
    for cyc in cycles:
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i+1)%len(cyc)]
            e = (u, v) if u < v else (v, u)
            cycle_edge_set.add(e)
            
    # All other edges are single edges initially
    single_edges = set()
    for e in edges:
        e_norm = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        if e_norm not in cycle_edge_set:
            single_edges.add(e_norm)
            
    # Now, try to merge cycles using 2-opt.
    # Also, try to extend cycles using single edges?
    # If we have a cycle C and a single edge (u, v), and u is in C, we can't just add it.
    # We need to break the cycle at u and insert v?
    # If C is ...-a-u-b-..., and we have edge (u, v) and (v, b), we can make ...-a-u-v-b-...
    # This requires (v, b) to be an edge.
    # This is more complex.
    
    # Let's stick to merging cycles first.
    
    # Precompute adjacency for fast lookup
    # adj_orig: dict of sets
    
    # For each pair of cycles, check if they can be merged.
    # Number of cycles is small (at most n/3).
    
    # To make it efficient, we can iterate until no more merges are possible.
    
    # A merge is possible between C1 and C2 if there exist edges e1=(u1, v1) in C1 and e2=(u2, v2) in C2
    # such that (u1, u2) in E and (v1, v2) in E.
    
    # We can represent a cycle by its edges.
    
    # Let's use a simple loop.
    
    changed = True
    while changed:
        changed = False
        # List of cycles
        # We need to update the list if we merge.
        
        # To avoid modifying list while iterating, we can collect merges first.
        # Or just restart the loop if a merge is found.
        
        # For simplicity, let's just find ONE merge and repeat.
        # Since n is small, this is fine.
        
        found_merge = False
        for i in range(len(cycles)):
            if found_merge: break
            for j in range(i+1, len(cycles)):
                if found_merge: break
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Get edges in c1
                # To allow rotation, we can fix c1[0] as start.
                # But we need to check all pairs of edges.
                
                # For each edge in c1: (u1, v1)
                # For each edge in c2: (u2, v2)
                # Check if (u1, u2) and (v1, v2) are in E
                # OR (u1, v2) and (v1, u2) are in E
                
                merged = False
                new_cycle = None
                
                # To be efficient, we can precompute neighbors in E for each vertex.
                # But let's just check directly.
                
                for k in range(len(c1)):
                    u1 = c1[k]
                    v1 = c1[(k+1)%len(c1)]
                    for l in range(len(c2)):
                        u2 = c2[l]
                        v2 = c2[(l+1)%len(c2)]
                        
                        # Check swap 1
                        e1 = (u1, u2) if u1 < u2 else (u2, u1)
                        e2 = (v1, v2) if v1 < v2 else (v2, v1)
                        if e1 in edge_index and e2 in edge_index:
                            # Merge
                            # New cycle: c1[0..k] + c2[0..l] reversed?
                            # Let's trace:
                            # c1: ... u1 -> v1 ...
                            # c2: ... u2 -> v2 ...
                            # Swap: u1 -> u2, v1 -> v2
                            # Path in c1 from v1 to u1 (forward)
                            # Path in c2 from u2 to v2 (forward)
                            # Wait, direction matters.
                            # c1 is a list. u1 is at k, v1 is at k+1.
                            # c2 is a list. u2 is at l, v2 is at l+1.
                            
                            # Segment 1 of c1: v1, ..., u1
                            # Segment 2 of c2: u2, ..., v2
                            
                            # We need to construct the new cycle.
                            # Let's rotate c1 so u1 is at 0? No, just slice.
                            
                            # c1 part: from v1 to u1 (wrapping around if needed)
                            # c1 indices: k+1, k+2, ..., n1-1, 0, ..., k
                            # c2 part: from u2 to v2 (wrapping around if needed)
                            # c2 indices: l, l+1, ..., n2-1, 0, ..., l+1
                            
                            # Actually, it's easier to just extract the two paths.
                            
                            # Path in c1 from v1 to u1:
                            # Start at index (k+1)%n1, end at index k.
                            p1 = []
                            idx = (k+1) % len(c1)
                            while True:
                                p1.append(c1[idx])
                                if idx == k: break
                                idx = (idx + 1) % len(c1)
                                
                            # Path in c2 from u2 to v2:
                            # Start at index l, end at index (l+1)%n2.
                            p2 = []
                            idx = l
                            while True:
                                p2.append(c2[idx])
                                if idx == (l+1) % len(c2): break
                                idx = (idx + 1) % len(c2)
                                
                            # New cycle: p1 + p2
                            new_cycle = p1 + p2
                            
                            # Verify:
                            # Last of p1 is u1. First of p2 is u2. Edge (u1, u2) exists.
                            # Last of p2 is v2. First of p1 is v1. Edge (v2, v1) exists.
                            
                            # Update cycles list
                            cycles[i] = new_cycle
                            del cycles[j]
                            found_merge = True
                            changed = True
                            break
                        if found_merge: break
                        
                        # Check swap 2
                        e1 = (u1, v2) if u1 < v2 else (v2, u1)
                        e2 = (v1, u2) if v1 < u2 else (u2, v1)
                        if e1 in edge_index and e2 in edge_index:
                            # Swap: u1 -> v2, v1 -> u2
                            # Path in c1 from v1 to u1
                            p1 = []
                            idx = (k+1) % len(c1)
                            while True:
                                p1.append(c1[idx])
                                if idx == k: break
                                idx = (idx + 1) % len(c1)
                                
                            # Path in c2 from v2 to u2 (REVERSED)
                            # c2: ... u2 -> v2 ...
                            # We want v2 -> ... -> u2
                            p2 = []
                            idx = (l+1) % len(c2)
                            while True:
                                p2.append(c2[idx])
                                if idx == l: break
                                idx = (idx - 1) % len(c2)
                                
                            new_cycle = p1 + p2
                            
                            cycles[i] = new_cycle
                            del cycles[j]
                            found_merge = True
                            changed = True
                            break
                        if found_merge: break
                if found_merge: break
            if found_merge: break
            
        # If no merge found, we can try to extend cycles with single edges?
        # This is harder. Let's skip for now.
        
    # Final pieces
    pieces = []
    for cyc in cycles:
        if len(cyc) >= 3:
            pieces.append(cyc)
            
    for e in single_edges:
        pieces.append([e[0], e[1]])
        
    return pieces

def greedy_decompose(edges, n):
    # Fallback
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    adj = {u: set() for u in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining:
        start = None
        for u in range(n):
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
            
        visited = {start}
        path = [start]
        stack = [(start, iter(sorted(adj[start])))]
        found = False
        while stack and not found:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in remaining:
                    continue
                if nb == start and len(path) >= 3:
                    pieces.append(path[:])
                    for i in range(len(path)):
                        u = path[i]
                        v = path[(i+1)%len(path)]
                        ee = (u, v) if u < v else (v, u)
                        remaining.discard(ee)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    found = True
                    break
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(sorted(adj[nb]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    visited.discard(path.pop())
                    
        if not found:
            break
            
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        # Allocate time based on remaining budget
        elapsed = time.time() - start_time
        remaining_time = 140 - elapsed
        per_graph_time = remaining_time / (3 - len(results))
        
        pieces = decompose_graph_ilp(edges, n, time_limit=per_graph_time)
        results.append(pieces)
        
        if time.time() - start_time > 145:
            # Fallback
            for spec in specs[specs.index((n,p,seed))+1:]:
                e = instance(*spec)
                results.append([list(ed) for ed in e])
            break
            
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
