import time
import random

N = 20
C = 20
DEADLINE = 145.0  # Safety margin

def _validate(edges):
    """Return True iff every 4-cycle is rainbow (4 distinct colours)."""
    if not edges:
        return True
    
    color = [[-1] * N for _ in range(N)]
    adj_mask = [0] * N
    for u, v, c in edges:
        if u == v:
            return False
        color[u][v] = c
        color[v][u] = c
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Check all 4-cycles
    # A 4-cycle is u-v-w-x-u
    # u < v < w < x is not sufficient to enumerate all.
    # Better: iterate u < v, then common neighbors w, x.
    for u in range(N):
        for v in range(u + 1, N):
            if color[u][v] == -1:
                continue
            # common neighbors of u and v
            common = adj_mask[u] & adj_mask[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            # Get list of common neighbors
            commons = []
            m = common
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                commons.append(i)
                m ^= lsb
                
            # Pairs of common neighbors
            for i in range(len(commons)):
                w = commons[i]
                for j in range(i + 1, len(commons)):
                    x = commons[j]
                    # Cycle u - w - v - x - u
                    c1 = color[u][w]
                    c2 = color[w][v]
                    c3 = color[v][x]
                    c4 = color[x][u]
                    if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                        continue # Should not happen if w,x are common neighbors
                    if len({c1, c2, c3, c4}) != 4:
                        return False
    return True

def construct():
    start_time = time.time()
    best_edges = []
    best_len = 0
    
    # Pre-compute all pairs
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    def try_run(seed):
        rng = random.Random(seed)
        
        color = [[-1] * N for _ in range(N)]
        adj_mask = [0] * N
        edges = []
        
        # Helper to check if adding (u,v,c) is valid
        def can_add(u, v, c):
            # Check cycles containing (u,v): u-v-x-y-u
            # x in N(v), y in N(u), x!=u, y!=v, x!=y, edge(x,y) exists
            # Colors: c, color[v][x], color[x][y], color[y][u] must be distinct
            
            # Iterate x in N(v)
            m_v = adj_mask[v]
            m_v &= ~(1 << u) # Exclude u
            while m_v:
                lsb_x = m_v & -m_v
                x = lsb_x.bit_length() - 1
                m_v ^= lsb_x
                
                # Iterate y in N(u)
                m_u = adj_mask[u]
                m_u &= ~(1 << v) # Exclude v
                while m_u:
                    lsb_y = m_u & -m_u
                    y = lsb_y.bit_length() - 1
                    m_u ^= lsb_y
                    
                    if x == y:
                        continue
                    if (adj_mask[x] & (1 << y)) == 0:
                        continue
                    
                    c1 = c
                    c2 = color[v][x]
                    c3 = color[x][y]
                    c4 = color[y][u]
                    
                    if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                        return False
            return True

        def add_edge(u, v, c):
            color[u][v] = c
            color[v][u] = c
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v, c])

        # Greedy construction
        order = all_pairs[:]
        rng.shuffle(order)
        
        for u, v in order:
            if time.time() > DEADLINE:
                break
            
            # Try colors in random order
            cols = list(range(C))
            rng.shuffle(cols)
            
            added = False
            for c in cols:
                if can_add(u, v, c):
                    add_edge(u, v, c)
                    added = True
                    break
            
            # If not added, we could try to "repair" by changing other colors, 
            # but for simplicity and speed, we just skip.
            
        return edges

    seed_counter = 0
    while time.time() < DEADLINE:
        edges = try_run(seed_counter)
        seed_counter += 1
        
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges
        
        # Optional: If we find a very dense graph, we can stop early
        if best_len >= 100: # Just a guess, max is 190
             # We can continue to try to improve, but 100 is already good
             pass

    return best_edges

if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
