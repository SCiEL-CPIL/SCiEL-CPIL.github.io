import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    adj = [0] * n

    def is_valid_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check if u and v have > 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # Check if adding (u,v) creates a K2,3 involving u and some other vertex w
        # A K2,3 exists if there are two vertices x,y with >=3 common neighbors.
        # Adding (u,v) affects pairs involving u or v.
        
        # 1. Pairs (u, w) for w in N(v) or w in N(u)?
        # If we add (u,v), then for any w, the common neighbors of (u,w) might change.
        # Specifically, v becomes a neighbor of u.
        # So for any w != v, v is now in N(u).
        # If v is also in N(w), then v is a common neighbor of (u,w).
        # We need to ensure that for all w, |N(u) \cap N(w)| <= 2.
        
        # Let's check all w in N(u) and w in N(v) and w not u,v.
        # Actually, it's enough to check w in N(u) and w in N(v) because those are the ones where the intersection changes.
        # For w not in N(u) and w not in N(v), the intersection doesn't change.
        # For w in N(u) but not N(v), v is in N(u) but not N(w) (unless w=v, excluded). So v is not common.
        # Wait, if w in N(u), and we add (u,v), does v become a common neighbor of (u,w)?
        # Only if v in N(w).
        
        # So we need to check:
        # For each w in N(u):
        #   If v in N(w), then v is a new common neighbor of (u,w).
        #   We need (adj[u] | (1<<v)) & adj[w] to have bit_count <= 2.
        #   Note: adj[u] already has v? No, we are checking before adding.
        #   So new_N_u = adj[u] | (1<<v).
        #   Check (new_N_u & adj[w]).bit_count() <= 2.
        
        # Similarly for w in N(v):
        #   If u in N(w), then u is a new common neighbor of (v,w).
        #   Check (adj[v] | (1<<u)) & adj[w] has bit_count <= 2.
        
        # Optimization:
        # Only w in N(u) such that v in N(w) are candidates for violation on the u-side.
        # Only w in N(v) such that u in N(w) are candidates for violation on the v-side.
        
        # Let's just iterate through all w in N(u) and N(v).
        
        new_u = adj[u] | (1 << v)
        new_v = adj[v] | (1 << u)
        
        # Check neighbors of u
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if w == v:
                temp ^= lsb
                continue
            # Check if (u,w) is still valid
            # Common neighbors of u and w in new graph:
            # (new_u & adj[w]).bit_count() must be <= 2
            if (new_u & adj[w]).bit_count() > 2:
                return False
            temp ^= lsb
            
        # Check neighbors of v
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if w == u:
                temp ^= lsb
                continue
            # Check if (v,w) is still valid
            # Common neighbors of v and w in new graph:
            # (new_v & adj[w]).bit_count() must be <= 2
            if (new_v & adj[w]).bit_count() > 2:
                return False
            temp ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_count = 0
    best_adj = [0] * n

    def greedy_init():
        nonlocal adj
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(u, v):
                add_edge(u, v)

    def polish():
        nonlocal adj
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True

    def unblocking_search():
        nonlocal adj
        # Try to unblock missing edges by swapping
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            missing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                # Check if (u,v) is blocked
                common = adj[u] & adj[v]
                if common.bit_count() <= 2:
                    # It's not blocked by common neighbors, try to add
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True
                        break # Restart loop as graph changed
                else:
                    # It is blocked. Try to remove one common neighbor edge to unblock.
                    # We need to remove (u, c) or (v, c) for c in common.
                    # Removing (u, c) reduces common neighbors by 1 (c is no longer in N(u)).
                    # But we must ensure the graph remains valid after removal and addition.
                    # Actually, if we remove (u,c), then (u,v) might become addable.
                    # Let's try each c in common.
                    temp = common
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        temp ^= lsb
                        
                        # Try swap: remove (u,c), add (u,v)
                        # We need to check if removing (u,c) is safe (it is, it's an edge)
                        # and if adding (u,v) is safe in the state where (u,c) is removed.
                        
                        # State: adj with (u,c) removed.
                        # Let's simulate.
                        # new_adj_u = adj[u] & ~(1<<c)
                        # new_adj_c = adj[c] & ~(1<<u)
                        # Others unchanged.
                        
                        # Check if (u,v) can be added in this state.
                        # Condition 1: (new_adj_u & adj[v]).bit_count() <= 2
                        # Condition 2: For all w in N(u) (in new state), (new_adj_u | (1<<v)) & adj[w] <= 2
                        # Condition 3: For all w in N(v), (adj[v] | (1<<u)) & adj[w] <= 2
                        
                        new_adj_u = adj[u] & ~(1 << c)
                        
                        if (new_adj_u & adj[v]).bit_count() > 2:
                            continue
                            
                        # Check other constraints
                        valid_swap = True
                        # Check neighbors of u in new state
                        temp2 = new_adj_u
                        while temp2:
                            lsb2 = temp2 & -temp2
                            w = lsb2.bit_length() - 1
                            temp2 ^= lsb2
                            if (new_adj_u | (1 << v)) & adj[w] and ((new_adj_u | (1 << v)) & adj[w]).bit_count() > 2:
                                valid_swap = False
                                break
                        if not valid_swap:
                            continue
                            
                        # Check neighbors of v
                        temp3 = adj[v]
                        while temp3:
                            lsb3 = temp3 & -temp3
                            w = lsb3.bit_length() - 1
                            temp3 ^= lsb3
                            if (adj[v] | (1 << u)) & adj[w] and ((adj[v] | (1 << u)) & adj[w]).bit_count() > 2:
                                valid_swap = False
                                break
                        if not valid_swap:
                            continue
                            
                        # Perform swap
                        remove_edge(u, c)
                        add_edge(u, v)
                        improved = True
                        break # Restart loop
                        
                    if improved:
                        break

    def run_sa():
        nonlocal adj, best_count, best_adj
        greedy_init()
        polish()

        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

        T = 1.0
        alpha = 0.99995
        min_T = 0.01

        while time.time() < deadline - 5:
            # Try unblocking first
            unblocking_search()
            
            cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
            if cur_count > best_count:
                best_count = cur_count
                best_adj = adj[:]
                
            # Then try random moves to escape local optima
            present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]

            move_type = random.random()
            if move_type < 0.5 and missing:
                u, v = random.choice(missing)
                if is_valid_add(u, v):
                    add_edge(u, v)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = adj[:]
            elif move_type < 0.8 and present:
                u, v = random.choice(present)
                if random.random() < math.exp(-1.0 / max(T, min_T)):
                    remove_edge(u, v)
                    cur_count -= 1
            else:
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    remove_edge(u1, v1)
                    if is_valid_add(u2, v2):
                        add_edge(u2, v2)
                    else:
                        add_edge(u1, v1)

            T = max(T * alpha, min_T)

        polish()
        unblocking_search()
        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

    # Run multiple SA restarts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        run_sa()
        cur_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
