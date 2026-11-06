import time
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        if m == 0:
            return 0, [], [], []
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # conflict[i] is the set of edges j that cannot share a color with i
        conflict = [set() for _ in range(m)]
        # For speed, we precompute the "forbidden vertices" for each edge
        # and then map vertices to incident edges
        inc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].add(i)
            inc[v].add(i)
            
        for i, (u, v) in enumerate(edges):
            # An edge j conflicts with i if it shares a vertex with i 
            # OR if it is adjacent to a vertex of i (i.e., shares a vertex with adj[u] or adj[v])
            # Actually, induced matching definition:
            # 1. No shared vertices: j must not touch u or v.
            # 2. No edge between endpoints: if j=(x,y), then (x,y) must not be in E.
            #    This means x and y must not be adjacent.
            #    Equivalently, x must not be in adj[y] and y must not be in adj[x].
            #    Wait, the conflict graph definition in the prompt says:
            #    "conflict[e] = edges f != e with an endpoint in adj[u] | adj[v]"
            #    Let's verify this.
            #    If f=(x,y) has x in adj[u], then edge (u,x) exists.
            #    Since e=(u,v) and f=(x,y) share no vertices (implied by induced matching? No, induced matching requires no shared vertices AND no connecting edges).
            #    If x is in adj[u], then x is connected to u.
            #    If f shares a vertex with e, they conflict.
            #    If f does not share a vertex with e, but x is in adj[u], then there is an edge (u,x).
            #    Does (u,x) connect e and f? 
            #    e has endpoints u,v. f has endpoints x,y.
            #    Edge (u,x) connects u (from e) to x (from f).
            #    So yes, f conflicts with e.
            #    So the set of conflicting edges for e=(u,v) is:
            #    {f | f shares a vertex with e} U {f | f has an endpoint in adj[u] U adj[v]}
            #    Note: If f shares a vertex with e, say x=u, then x is in {u,v} which is subset of adj[u]Uadj[v]? No.
            #    But if x=u, then u is in adj[u]? No.
            #    However, if x=u, then f touches u.
            #    The condition "endpoint in adj[u] | adj[v]" covers the "connected by an edge" case.
            #    We must also include edges sharing a vertex.
            #    Let's stick to the prompt's definition which seems to imply the union.
            #    Actually, if f shares a vertex u with e=(u,v), then f=(u,y).
            #    Is u in adj[u] U adj[v]? Not necessarily.
            #    But f conflicts with e because they share a vertex.
            #    The prompt's definition: "conflict[e] = edges f != e with an endpoint in adj[u] | adj[v]"
            #    If this definition is correct, it implies that sharing a vertex is NOT captured by "endpoint in adj[u]|adj[v]" unless the graph has self-loops or something.
            #    Wait, if f=(u,y), endpoint u. Is u in adj[u]? No. Is u in adj[v]? Only if (u,v) is in adj[v], i.e., if v is adjacent to u. Yes, (u,v) is an edge, so u is in adj[v].
            #    So if f shares vertex u with e=(u,v), then u is in adj[v] (since v is adjacent to u).
            #    So f has an endpoint u which is in adj[v]. Thus f is in conflict.
            #    So the prompt's definition is correct and complete.
            
            forb = adj[u] | adj[v]
            s = set()
            for w in forb:
                s |= inc[w]
            s.discard(i)
            conflict[i] = s
        return n, adj, inc, conflict

    def find_maximal_induced_matching(edges, conflict, start_edge, rng):
        """
        Greedily grow an induced matching starting with start_edge.
        Returns a set of edge indices.
        """
        m = len(edges)
        current = {start_edge}
        # We maintain a set of "available" edges that are compatible with current
        # Initially, available = all edges j such that j is not in conflict[start_edge] and j != start_edge
        available = set(range(m))
        available.discard(start_edge)
        available -= conflict[start_edge]
        
        # To make it faster, we can just iterate through a shuffled list of candidates
        # and add if compatible with all in current.
        # But checking compatibility with all in current is O(|current| * |conflict|).
        # Better: maintain a set of 'blocked' edges.
        # When we add edge e to current, all edges in conflict[e] become blocked.
        blocked = set(conflict[start_edge])
        
        # We want to maximize the size. A simple greedy is to pick the edge with fewest conflicts among available?
        # Or just pick randomly from available?
        # Let's try picking the edge with minimum degree in the "available" subgraph?
        # That might be slow.
        # Let's just pick randomly from available, but bias towards edges that block fewer other edges?
        # Actually, for set cover, we want large sets.
        # So we want a large induced matching.
        # Heuristic: pick the edge that conflicts with the fewest other available edges.
        
        # This might be slow if we do it for every step.
        # Given m is small (~100-200), O(m^2) per matching construction is fine.
        # Number of matchings needed is ~m. Total O(m^3) ~ 8e6 ops. Fast.
        
        while available:
            # Find edge in available with minimum number of conflicts with other available edges
            # This is a greedy heuristic for Max Independent Set on the conflict graph.
            best_edge = -1
            min_conflicts = m + 1
            for e in available:
                # Count how many edges in available are in conflict[e]
                cnt = 0
                for f in available:
                    if f != e and f in conflict[e]:
                        cnt += 1
                if cnt < min_conflicts:
                    min_conflicts = cnt
                    best_edge = e
                    if cnt == 0:
                        break # Can't do better than 0
            
            if best_edge == -1:
                break
                
            # Add best_edge to current
            current.add(best_edge)
            available.discard(best_edge)
            
            # Update blocked
            new_blocked = conflict[best_edge]
            blocked |= new_blocked
            
            # Remove newly blocked edges from available
            # Also remove best_edge's conflicts from available
            available -= new_blocked
            
        return current

    def solve_instance(edges, deadline):
        m = len(edges)
        if m == 0:
            return []
        
        n, adj, inc, conflict = build_conflict(edges)
        rng = Random(42)
        
        # 1. Generate a pool of induced matchings (cliques in the conflict graph's complement)
        # We want a diverse set of large induced matchings.
        matchings = []
        seen_matchings = set()
        
        # Strategy: 
        # - Generate some random maximal induced matchings.
        # - Generate some greedy "minimum conflict" induced matchings.
        
        # We will generate up to 50 matchings or until time runs out.
        t_match = time.time() + 5.0
        count = 0
        while time.time() < t_match and count < 50:
            # Pick a random starting edge
            start = rng.randrange(m)
            matching = find_maximal_induced_matching(edges, conflict, start, rng)
            sig = frozenset(matching)
            if sig not in seen_matchings:
                seen_matchings.add(sig)
                matchings.append(matching)
            count += 1
            
        if not matchings:
            # Fallback: single edge matchings
            matchings = [ {i} for i in range(m) ]
            
        # 2. Set Cover: Select a subset of matchings that covers all edges with minimum count.
        # Greedy Set Cover:
        # At each step, pick the matching that covers the most uncovered edges.
        
        uncovered = set(range(m))
        selected_matchings = []
        
        # Precompute coverage counts for each matching?
        # Since matchings are static, we can just calculate on the fly.
        
        # To speed up, we can maintain a count of how many times each edge is covered by remaining matchings?
        # No, we just need to pick the one with max intersection with 'uncovered'.
        
        # Optimization: Sort matchings by size initially?
        # Or just iterate.
        
        # Since m is small, we can just do a simple greedy.
        # To add randomness/diversity, we might run this greedy multiple times with slight perturbations?
        # Let's run one standard greedy first.
        
        avail_matchings = matchings[:]
        while uncovered:
            best_match = None
            best_count = -1
            # Shuffle to break ties randomly
            rng.shuffle(avail_matchings)
            for match in avail_matchings:
                # Count how many edges in match are in uncovered
                cnt = 0
                for e in match:
                    if e in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_match = match
                if cnt == len(match):
                    break # Perfect coverage of this matching
            
            if best_match is None or best_count == 0:
                # Should not happen if we have single-edge matchings
                break
                
            selected_matchings.append(best_match)
            uncovered -= best_match
            avail_matchings.remove(best_match)
            
        # Initial solution: list of edge lists
        initial_solution = [list(m) for m in selected_matchings]
        
        # 3. Local Search: Merge and Move
        # Representation: list of sets of edge indices.
        # We want to minimize the number of sets.
        
        solution = [set(m) for m in selected_matchings]
        
        def is_valid_solution(sol):
            # Check if any two edges in the same set conflict
            for s in sol:
                for e in s:
                    for f in s:
                        if e != f and f in conflict[e]:
                            return False
            # Check if all edges are covered
            covered = set()
            for s in sol:
                covered |= s
            return covered == set(range(m))
        
        # Local Search Loop
        t_ls = time.time()
        ls_deadline = deadline - 1.0
        
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < ls_deadline and no_improve < max_no_improve:
            improved = False
            
            # Try to merge two sets
            # For each pair of sets A, B, check if they can be merged.
            # They can be merged if no edge in A conflicts with any edge in B.
            # i.e., for all a in A, b in B, b not in conflict[a].
            
            # Optimization: Precompute "compatibility" between sets?
            # Or just check on the fly.
            
            # To find a merge quickly, we can iterate through pairs.
            # Number of sets is small (~20-50).
            
            merged = False
            # Shuffle order to avoid bias
            indices = list(range(len(solution)))
            rng.shuffle(indices)
            
            for i_idx in indices:
                if merged: break
                A = solution[i_idx]
                if not A: continue
                for j_idx in indices:
                    if j_idx <= i_idx: continue
                    B = solution[j_idx]
                    if not B: continue
                    
                    # Check if A and B are compatible
                    compatible = True
                    for a in A:
                        for b in B:
                            if b in conflict[a]:
                                compatible = False
                                break
                        if not compatible: break
                    
                    if compatible:
                        # Merge B into A
                        A |= B
                        solution[j_idx] = set()
                        merged = True
                        improved = True
                        break
            
            if improved:
                # Remove empty sets
                solution = [s for s in solution if s]
                no_improve = 0
                continue
            
            no_improve += 1
            
            # If no merge, try to move an edge from one set to another to enable a future merge?
            # Or just try to "compact" sets?
            # A simple move: Move an edge e from set A to set B if e is compatible with B.
            # This doesn't reduce count, but might enable a merge later.
            
            # Let's try a "shake": pick a random edge, remove it, reinsert into a random compatible set.
            # This is a bit random.
            
            # Better: Try to find a set A and an edge e in A such that e can be moved to some B,
            # and then A and B can be merged?
            # That's complex.
            
            # Let's try a simple "edge move" that doesn't increase count, just rearranges.
            # This helps escape local minima in the "merge" space.
            
            moved = False
            indices = list(range(len(solution)))
            rng.shuffle(indices)
            
            for i_idx in indices:
                if moved: break
                A = solution[i_idx]
                if len(A) < 2: continue
                # Pick a random edge from A
                e = rng.choice(list(A))
                A.remove(e)
                
                # Try to put e in another set B
                for j_idx in indices:
                    if j_idx == i_idx: continue
                    B = solution[j_idx]
                    # Check if e is compatible with B
                    compatible = True
                    for b in B:
                        if b in conflict[e]:
                            compatible = False
                            break
                    if compatible:
                        B.add(e)
                        moved = True
                        break
                
                if not moved:
                    # Put it back
                    A.add(e)
            
            if moved:
                # Don't reset no_improve on moves, only on merges?
                # Or reset? Moves are "neutral" moves.
                # Let's not reset, so we don't loop forever on moves without merges.
                pass
        
        # Final validation and remapping
        if not is_valid_solution(solution):
            # Fallback to greedy coloring
            col = [-1] * m
            for i in range(m):
                used = set()
                for j in conflict[i]:
                    if col[j] >= 0:
                        used.add(col[j])
                c = 0
                while c in used:
                    c += 1
                col[i] = c
            # Remap to 0..k-1
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            return [remap[c] for c in col]
        
        # Remap solution sets to 0..k-1
        # We need to assign a color index to each set.
        color_of_edge = [-1] * m
        for idx, s in enumerate(solution):
            for e in s:
                color_of_edge[e] = idx
        
        # Remap indices to 0..k-1
        present = sorted(set(color_of_edge))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[c] for c in color_of_edge]

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    for idx, spec in enumerate(specs):
        edges = gen_instance(*spec)
        m = len(edges)
        left_after = len(specs) - 1 - idx
        inst_dl = min(DEADLINE - 3.0 - 2.0 * left_after, time.time() + 50.0)
        if inst_dl < time.time() + 5.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 5.0)
        
        try:
            out.append(solve_instance(edges, inst_dl))
        except Exception:
            # Fallback
            n, adj, inc, conflict = build_conflict(edges)
            col = [-1] * m
            for i in range(m):
                used = set()
                for j in conflict[i]:
                    if col[j] >= 0:
                        used.add(col[j])
                c = 0
                while c in used:
                    c += 1
                col[i] = c
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    return out
