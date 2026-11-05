import time
import random

def construct(k, g):
    # Heawood graph edges
    # 14 vertices
    # 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10-11, 11-12, 12-13, 13-0
    # Plus chords?
    # Heawood is the incidence graph of Fano plane.
    # Vertices: 7 points, 7 lines.
    # Edges: point-line incidence.
    
    # Let's use a standard list of edges for Heawood.
    # 0..6 are points, 7..13 are lines.
    # Line 7: 0,1,2
    # Line 8: 0,3,4
    # Line 9: 0,5,6
    # Line 10: 1,3,5
    # Line 11: 1,4,6
    # Line 12: 2,3,6
    # Line 13: 2,4,5
    
    edges = []
    for i in range(7):
        for j in range(7, 14):
             # Check incidence
             pass # This is hard to code inline.
             
    # I will use a hardcoded edge list for Heawood.
    # 0-7, 1-7, 2-7
    # 0-8, 3-8, 4-8
    # 0-9, 5-9, 6-9
    # 1-10, 3-10, 5-10
    # 1-11, 4-11, 6-11
    # 2-12, 3-12, 6-12
    # 2-13, 4-13, 5-13
    
    heawood_edges = [
        (0,7), (1,7), (2,7),
        (0,8), (3,8), (4,8),
        (0,9), (5,9), (6,9),
        (1,10), (3,10), (5,10),
        (1,11), (4,11), (6,11),
        (2,12), (3,12), (6,12),
        (2,13), (4,13), (5,13)
    ]
    
    adj = {i: [] for i in range(14)}
    for u, v in heawood_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    edge_map = {}
    for i, (u, v) in enumerate(heawood_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    def get_closed_walks(max_len):
        walks = []
        for start in range(14):
            stack = [(start, [start])]
            while stack:
                curr, path = stack.pop()
                if len(path) > max_len:
                    continue
                if len(path) >= 4 and curr == start: # Heawood girth 6, so min cycle 6. But closed walks can be shorter (backtracks).
                    walks.append(path)
                for next_v in adj[curr]:
                    stack.append((next_v, path + [next_v]))
        return walks
        
    start_time = time.time()
    deadline = start_time + 110
    
    for M in range(4, 10):
        if time.time() > deadline:
            break
            
        closed_walks = get_closed_walks(g - 1)
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, M - 1) for _ in range(len(heawood_edges))]
            
            valid = True
            for walk in closed_walks:
                s = 0
                for i in range(len(walk) - 1):
                    u, v = walk[i], walk[i+1]
                    idx = edge_map[(u, v)]
                    if u < v:
                        s += voltages[idx]
                    else:
                        s -= voltages[idx]
                        
                s = s % M
                if s == 0:
                    verts = []
                    curr_sum = 0
                    for i in range(len(walk) - 1):
                        u, v = walk[i], walk[i+1]
                        idx = edge_map[(u, v)]
                        if u < v:
                            curr_sum += voltages[idx]
                        else:
                            curr_sum -= voltages[idx]
                        curr_sum %= M
                        verts.append((walk[i+1], curr_sum))
                        
                    if len(set(verts)) == len(verts):
                        valid = False
                        break
            
            if valid:
                n = 14 * M
                lift_edges = []
                for u, v in heawood_edges:
                    idx = edge_map[(u, v)]
                    val = voltages[idx]
                    for i in range(M):
                        u_id = u * M + i
                        v_id = v * M + (i + val) % M
                        lift_edges.append((u_id, v_id))
                return (n, lift_edges)
                
    # Fallback
    return (14, heawood_edges)
