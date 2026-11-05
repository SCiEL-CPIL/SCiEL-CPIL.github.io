import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # 1. Construct Edges
    edges = set()

    # Base Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # Chords from 0 to m for m=2..20
    # These create cycles of length m+1 and 41-m
    for m in range(2, 21):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # 2. Construct Cycles
    cycles = []

    # Length 3 to 40
    for k in range(3, N + 1):
        if k == N:
            # Full cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Uses chord (0, k-1) and path 0-1-...-(k-1)
            # Cycle: 0-1-...-(k-1)-0
            cycles.append(list(range(k)))
        else:
            # Uses chord (0, m) where 41-m = k => m = 41-k
            # Path: 0-39-...-m
            m = 41 - k
            # Vertices: 0, 39, 38, ..., m
            cycle_verts = [0] + list(range(39, m - 1, -1))
            cycles.append(cycle_verts)

    edge_list = list(edges)

    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    # Self-test
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    for c in res['cycles']:
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c}")
                break
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
