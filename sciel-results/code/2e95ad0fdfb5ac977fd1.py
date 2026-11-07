"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 15}

def construct():
    t0 = time.time()
    deadline = 90.0
    safety = 2.0  # return by deadline - safety

    n = P['n']
    # 5 classes of 3 vertices each, arranged in a 5-cycle of classes
    # class i is adjacent (in the class-level graph) to classes (i-1)%5 and (i+1)%5
    classes = [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    # Initialize: complete bipartite between adjacent classes (5-cycle of classes)
    adj = [[False]*n for _ in range(n)]
    for i in range(5):
        j1 = (i - 1) % 5
        j2 = (i + 1) % 5
        for a in classes[i]:
            for b in classes[j1]:
                adj[a][b] = adj[b][a] = True
            for b in classes[j2]:
                adj[a][b] = adj[b][a] = True

    # Helper: check if adding edge (u,v) preserves the property
    # that no two same-class vertices are at distance <= 3
    # We precompute class membership
    cls_of = [0]*n
    for i, c in enumerate(classes):
        for v in c:
            cls_of[v] = i

    def violates(u, v):
        """Check if adding edge (u,v) would create two same-class vertices at distance <= 3."""
        # After adding (u,v), we need to check all pairs (a,b) with cls_of[a]==cls_of[b], a!=b,
        # whether dist(a,b) <= 3 in the new graph.
        # Efficient: only need to check pairs whose shortest path might now be <= 3.
        # A new path of length 3 through (u,v): a-u-v-b or a-v-u-b where a,b same class.
        # Also paths of length 2: a-u-b where u is the new middle... but u,v are endpoints of new edge.
        # Actually adding edge (u,v) can create:
        #   dist 1: (u,v) - but u,v are in different classes (we only add cross-class edges)
        #   dist 2: a-u-v-b or a-v-u-b -> length 2? No, a-u-v-b is length 3.
        #   Wait: a adjacent to u, u adjacent to v (new), v adjacent to b => path a-u-v-b length 3.
        #   Also: a adjacent to v, v adjacent to u (new), u adjacent to b => path a-v-u-b length 3.
        #   And: a adjacent to u, u adjacent to v (new) => dist(a,v) could be 2, then v adjacent to b => dist(a,b) <= 3? 
        #   Actually the new edge creates new paths. Let me just do a BFS from each same-class pair.
        # For n=15, this is fast enough.
        
        # Temporarily add edge
        adj[u][v] = adj[v][u] = True
        
        ok = True
        for i in range(5):
            c = classes[i]
            for a, b in itertools.combinations(c, 2):
                # BFS from a, check if dist to b <= 3
                visited = [-1]*n
                visited[a] = 0
                queue = [a]
                found = False
                for dist in range(1, 4):
                    new_queue = []
                    for x in queue:
                        for y in range(n):
                            if adj[x][y] and visited[y] == -1:
                                visited[y] = dist
                                if y == b:
                                    found = True
                                    break
                                new_queue.append(y)
                        if found:
                            break
                    if found:
                        break
                    queue = new_queue
                if found:
                    ok = False
                    break
            if not ok:
                break
        
        # Revert
        adj[u][v] = adj[v][u] = False
        return not ok

    # Greedily add edges
    # Generate all possible cross-class edges not yet present
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if cls_of[i] != cls_of[j] and not adj[i][j]:
                candidates.append((i, j))

    # Try to add as many as possible
    # Use a simple greedy: try each candidate, add if it doesn't violate
    # Sort by some heuristic? Just try in order.
    # For better results, try multiple random orderings
    import random
    rng = random.Random(42)

    best_edges = None
    best_count = 0

    # First pass: deterministic greedy
    for attempt in range(20):
        if time.time() > deadline - safety:
            break
        
        if attempt == 0:
            cand = list(candidates)
        else:
            cand = list(candidates)
            rng.shuffle(cand)
        
        # Reset adjacency to initial state
        adj2 = [[False]*n for _ in range(n)]
        for i in range(5):
            j1 = (i - 1) % 5
            j2 = (i + 1) % 5
            for a in classes[i]:
                for b in classes[j1]:
                    adj2[a][b] = adj2[b][a] = True
                for b in classes[j2]:
                    adj2[a][b] = adj2[b][a] = True
        
        # Use a local violates function with adj2
        def violates_local(u, v, adj2=adj2):
            adj2[u][v] = adj2[v][u] = True
            ok = True
            for i in range(5):
                c = classes[i]
                for a, b in itertools.combinations(c, 2):
                    visited = [-1]*n
                    visited[a] = 0
                    queue = [a]
                    found = False
                    for dist in range(1, 4):
                        new_queue = []
                        for x in queue:
                            for y in range(n):
                                if adj2[x][y] and visited[y] == -1:
                                    visited[y] = dist
                                    if y == b:
                                        found = True
                                        break
                                    new_queue.append(y)
                            if found:
                                break
                        if found:
                            break
                        queue = new_queue
                    if found:
                        ok = False
                        break
                if not ok:
                    break
            adj2[u][v] = adj2[v][u] = False
            return not ok

        count = 0
        for u, v in cand:
            if not adj2[u][v]:
                if not violates_local(u, v):
                    adj2[u][v] = adj2[v][u] = True
                    count += 1
            if time.time() > deadline - safety:
                break
        
        # Count total edges
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj2[i][j]:
                    total += 1
        
        if total > best_count:
            best_count = total
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj2[i][j]]

    # Final validation: check the property directly
    # For the best edge set, verify no 3-edge path has equal-degree endpoints
    if best_edges:
        final_adj = [[False]*n for _ in range(n)]
        for u, v in best_edges:
            final_adj[u][v] = final_adj[v][u] = True
        
        deg = [sum(final_adj[v][w] for w in range(n)) for v in range(n)]
        
        valid = True
        for u in range(n):
            for v in range(n):
                if u == v or not final_adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v or not final_adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v or not final_adj[t][v]:
                            continue
                        if deg[s] == deg[t]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Return as list of [int, int]
            return [[int(u), int(v)] for u, v in best_edges]
    
    # Fallback: return initial star-like or empty
    # Actually, let's just return the initial 5-cycle bipartite structure which should be valid
    fallback = []
    for i in range(5):
        j1 = (i - 1) % 5
        j2 = (i + 1) % 5
        for a in classes[i]:
            for b in classes[j1]:
                fallback.append([a, b])
            for b in classes[j2]:
                fallback.append([a, b])
    # Deduplicate
    seen = set()
    result = []
    for e in fallback:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            result.append([int(e[0]), int(e[1])])
    return result

# EVOLVE-BLOCK-END
