import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 15
    # Initialize with K_{7,8}
    # Partition A: 0..6 (size 7)
    # Partition B: 7..14 (size 8)
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    for u in range(7):
        for v in range(7, 15):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            
    best_edges = [list(e) for e in edges]
    best_count = len(edges)
    
    def update_best():
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
    def can_add(u, v):
        """
        Check if adding edge (u, v) to current graph (adj, deg) keeps it valid.
        Assumes (u, v) is NOT in adj.
        """
        if u == v:
            return False
        if v in adj[u]:
            return True # Already present, no change needed? 
                        # But we call this for missing edges.
        
        du = deg[u]
        dv = deg[v]
        
        # New degrees
        du_new = du + 1
        dv_new = dv + 1
        
        # 1. Paths s-u-v-t
        # s in N(u) \ {v}
        # t in N(v) \ {u, s}
        # Check deg[s] != deg[t]
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # 2. Paths s-v-u-t
        # s in N(v) \ {u}
        # t in N(u) \ {v, s}
        # Check deg[s] != deg[t]
        for s in adj[v]:
            if s == u:
                continue
            ds = deg[s]
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # 3. Paths u-v-x-y
        # v in N(u) (in new graph)
        # x in N(v) \ {u}
        # y in N(x) \ {v, u}
        # Endpoints u, y. Check deg_new[u] != deg[y]
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == v or y == u:
                    continue
                if du_new == deg[y]:
                    return False
        
        # 4. Paths v-u-x-y
        # u in N(v) (in new graph)
        # x in N(u) \ {v}
        # y in N(x) \ {u, v}
        # Endpoints v, y. Check deg_new[v] != deg[y]
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if dv_new == deg[y]:
                    return False
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append((u, v))

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove((u, v))

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    present = set(edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add missing edges
        # Randomize order to avoid bias
        missing = [p for p in all_pairs if p not in present]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if can_add(u, v):
                add_edge(u, v)
                present.add((u, v))
                update_best()
                improved = True
                # Restart missing list calculation? Or just continue?
                # Continuing is fine, but the list `missing` might contain newly added edges?
                # No, we only add from `missing`.
                # However, adding an edge might invalidate other potential additions?
                # No, we only add if valid.
                # But adding an edge changes degrees, which might make previously "safe" edges unsafe?
                # We check `can_add` dynamically.
                # So we should probably re-evaluate.
                # But for speed, we just continue with the current list.
                # If an edge becomes unsafe, `can_add` will return False.
        
        if time.time() > deadline - 10:
            break
            
        # Phase 2: If no edges added, try swaps
        # Try to remove an edge and add a different one
        # This is expensive: O(E * E_missing)
        # We limit the number of attempts.
        
        if not improved:
            # Try a few random swaps
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                if not edges:
                    break
                e_rm = random.choice(edges)
                u_rm, v_rm = e_rm
                remove_edge(u_rm, v_rm)
                present.discard(e_rm)
                
                # Try to add any missing edge
                found_swap = False
                missing_now = [p for p in all_pairs if p not in present]
                random.shuffle(missing_now)
                
                for u_add, v_add in missing_now:
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        present.add((u_add, v_add))
                        found_swap = True
                        # We swapped, edge count is same.
                        # But this might open up possibilities for future additions.
                        # So we break and go back to Phase 1.
                        break
                
                if found_swap:
                    improved = True
                    break
                else:
                    # Revert
                    add_edge(u_rm, v_rm)
                    present.add(e_rm)
                    
    return best_edges

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
