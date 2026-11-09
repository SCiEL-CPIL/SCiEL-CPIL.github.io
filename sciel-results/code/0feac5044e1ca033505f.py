import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

edges = []
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))

cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, c)
                e3 = edge_idx(c, d)
                e4 = edge_idx(d, a)
                cycles.append((e1, e2, e3, e4))
                
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, d)
                e3 = edge_idx(d, c)
                e4 = edge_idx(c, a)
                cycles.append((e1, e2, e3, e4))
                
                e1 = edge_idx(a, c)
                e2 = edge_idx(c, b)
                e3 = edge_idx(b, d)
                e4 = edge_idx(d, a)
                cycles.append((e1, e2, e3, e4))

edge_cycles = [[] for _ in range(NUM_EDGES)]
for cyc in cycles:
    for pos in range(4):
        e = cyc[pos]
        others = [cyc[(pos + 1) % 4], cyc[(pos + 2) % 4], cyc[(pos + 3) % 4]]
        edge_cycles[e].append(others)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    colors = [0] * NUM_EDGES
    next_color = 1
    
    def can_recolor(e, c_e):
        for others in edge_cycles[e]:
            c2 = colors[others[0]]
            c3 = colors[others[1]]
            c4 = colors[others[2]]
            
            count_match = 0
            if c2 == c_e: count_match += 1
            if c3 == c_e: count_match += 1
            if c4 == c_e: count_match += 1
            
            if count_match == 1:
                # Find the two non-matching
                vals = []
                if c2 != c_e: vals.append(c2)
                if c3 != c_e: vals.append(c3)
                if c4 != c_e: vals.append(c4)
                if vals[0] != vals[1]:
                    return False
            elif count_match == 0:
                if c2 != c3 and c3 != c4 and c2 != c4:
                    return False
        return True

    while True:
        if time.time() > deadline:
            break
        changed = False
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            c_e = colors[e]
            if can_recolor(e, c_e):
                colors[e] = next_color
                next_color += 1
                changed = True
        if not changed:
            break
            
    return colors
# EVOLVE-BLOCK-END
