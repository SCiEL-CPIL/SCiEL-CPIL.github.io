import time
import random
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    # 1. Base Structure: Hamiltonian Cycle
    # Edges: (i, (i+1)%N) for i in 0..N-1
    # This provides a cycle of length 40.
    # It also provides paths of any length L between any two vertices.
    # Specifically, the path from 0 to k (k<N) has length k.
    # The path from 0 to k (going the other way) has length N-k.
    
    # We need cycles of length 3, 4, ..., 39.
    # A cycle of length L can be formed if we have a chord (u, v) 
    # such that the distance along the base cycle between u and v is L or N-L.
    # If we fix u=0, we need chords (0, v) such that for every L in 3..39,
    # there exists a v where dist(0, v) is L or N-L.
    
    # Let's analyze the distances provided by chords from 0.
    # If we add chord (0, v), it creates cycles of length d and N-d, where d is dist(0, v) on the base cycle.
    # We need to cover all lengths 3..39.
    # Length 40 is covered by the base cycle itself.
    
    # Let S be the set of chords (0, v).
    # The set of cycle lengths provided by chords is {d, N-d | (0, v) in S}.
    # We need this set to cover {3, 4, ..., 39}.
    
    # Note: dist(0, v) can be 1, 2, ..., 39.
    # dist(0, 1) = 1 -> cycle length 1 (invalid, simple graph) and 39. 
    # Wait, a cycle must have at least 3 vertices.
    # If dist(0, v) = 1, the "cycle" is 0-1-0, which is not a simple cycle of length >= 3. 
    # Actually, the cycle formed by chord (0,1) and path 0-1 is length 2? No, simple cycle requires distinct vertices.
    # The cycle formed by chord (0, v) and the path of length d is a simple cycle of length d+1? 
    # No. The vertices are 0, v, and the intermediate nodes.
    # Path: 0 -> ... -> v. Length of path is d edges. 
    # Cycle: 0 -> ... -> v -> 0. Total edges = d + 1.
    # So a chord (0, v) at distance d creates a cycle of length d+1.
    # It also creates a cycle of length (N-d)+1 using the other path.
    
    # Let's re-verify.
    # Base cycle: 0-1-2-...-39-0.
    # Chord (0, k).
    # Path 1: 0-1-...-k. Edges: k. Cycle: 0-1-...-k-0. Length: k+1.
    # Path 2: 0-39-...-k. Edges: N-k. Cycle: 0-39-...-k-0. Length: N-k+1.
    
    # We need cycles of length 3, 4, ..., 39.
    # This corresponds to distances d such that d+1 in {3..39} or N-d+1 in {3..39}.
    # d+1 >= 3 => d >= 2.
    # d+1 <= 39 => d <= 38.
    # So we need chords (0, k) where k is in 2..38 such that the set of lengths {k+1, N-k+1} covers 3..39.
    
    # Let's find the minimal set of k's.
    # We need to cover lengths 3, 4, ..., 39.
    # Length 3: needs k=2 (2+1=3) or k=38 (40-38+1=3).
    # Length 4: needs k=3 (3+1=4) or k=37 (40-37+1=4).
    # ...
    # Length 20: needs k=19 (19+1=20) or k=21 (40-21+1=20).
    # Length 21: needs k=20 (20+1=21) or k=20 (40-20+1=21). Note k=20 gives both.
    
    # If we pick k=20, we get lengths 21 and 21. (Just 21).
    # If we pick k=19, we get lengths 20 and 22.
    # If we pick k=18, we get lengths 19 and 23.
    
    # Let's try to cover {3, ..., 39} with pairs {k+1, N-k+1}.
    # Let L be the target length.
    # We need k = L-1 or k = N-L+1.
    # Since k must be in 2..38 (to ensure simple cycle length >= 3 and <= 39? 
    # Wait, if k=1, length is 2 (invalid). If k=39, length is 40 (base cycle, already there).
    # So k in 2..38.
    
    # Let's check if we can cover all lengths with a small set of k's.
    # Each k covers 2 lengths (unless k=20, covers 1 unique length 21? No, 20+1=21, 40-20+1=21. Yes, just 21).
    # Actually, if k=20, the two paths are symmetric. Cycle length is 21.
    # So k=20 covers {21}.
    # k=19 covers {20, 22}.
    # k=18 covers {19, 23}.
    # k=17 covers {18, 24}.
    # k=16 covers {17, 25}.
    # k=15 covers {16, 26}.
    # k=14 covers {15, 27}.
    # k=13 covers {14, 28}.
    # k=12 covers {13, 29}.
    # k=11 covers {12, 30}.
    # k=10 covers {11, 31}.
    # k=9 covers {10, 32}.
    # k=8 covers {9, 33}.
    # k=7 covers {8, 34}.
    # k=6 covers {7, 35}.
    # k=5 covers {6, 36}.
    # k=4 covers {5, 37}.
    # k=3 covers {4, 38}.
    # k=2 covers {3, 39}.
    
    # We need to cover {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}.
    
    # Using the pairs above:
    # k=2 -> {3, 39}
    # k=3 -> {4, 38}
    # k=4 -> {5, 37}
    # k=5 -> {6, 36}
    # k=6 -> {7, 35}
    # k=7 -> {8, 34}
    # k=8 -> {9, 33}
    # k=9 -> {10, 32}
    # k=10 -> {11, 31}
    # k=11 -> {12, 30}
    # k=12 -> {13, 29}
    # k=13 -> {14, 28}
    # k=14 -> {15, 27}
    # k=15 -> {16, 26}
    # k=16 -> {17, 25}
    # k=17 -> {18, 24}
    # k=18 -> {19, 23}
    # k=19 -> {20, 22}
    # k=20 -> {21}
    
    # If we select k = 2, 3, ..., 20, we cover all lengths 3..39.
    # Number of chords: 19 (from 2 to 20 inclusive).
    # Total edges: 40 (base) + 19 (chords) = 59.
    # Score: 59 - 40 = 19.
    
    # Can we do better?
    # Each chord covers at most 2 lengths.
    # We have 37 lengths to cover (3..39).
    # Minimum chords needed: ceil(37/2) = 19.
    # So 19 chords is the theoretical minimum if we only use chords from 0.
    
    # Could we use chords not from 0?
    # A chord (u, v) creates cycles of length dist(u,v)+1 and N-dist(u,v)+1.
    # This is equivalent to the above logic, just rotated.
    # So 19 chords is likely the minimum for this "base cycle + chords" strategy.
    
    # Is it possible to have fewer than 19 chords?
    # Only if some cycles are formed by combinations of multiple chords?
    # No, the requirement is "one vertex-list cycle of every length".
    # The cycles must be simple.
    # If we don't use a base cycle, we might need more edges.
    # The base cycle is very efficient (40 edges for 1 length + paths).
    
    # Let's verify the construction.
    # Edges:
    # 1. (i, (i+1)%40) for i=0..39.
    # 2. (0, k) for k=2..20.
    
    # Cycles:
    # Length L in 3..39:
    # Find k such that L = k+1 or L = 40-k+1.
    # If L <= 21: k = L-1. Cycle: 0, 1, ..., L-1, 0.
    # If L > 21: k = 40-L+1. Cycle: 0, 39, ..., k, 0.
    # Note: For L=21, k=20. Cycle: 0, 1, ..., 20, 0.
    
    # Let's double check L=39.
    # k = 39-1 = 38. But we only have chords up to 20.
    # Wait. My previous analysis:
    # k=2 covers {3, 39}.
    # So for L=39, we use k=2.
    # Cycle: 0, 39, 2, 0? No.
    # Path from 0 to 2 the long way: 0-39-...-2. Length 38.
    # Cycle: 0-39-...-2-0. Length 39.
    # Vertices: 0, 39, 38, ..., 2.
    # Edges: (0,39), (39,38), ..., (3,2), (2,0).
    # All these edges exist?
    # (0,39) is in base cycle.
    # (3,2) is in base cycle.
    # (2,0) is the chord.
    # Yes.
    
    # So the set of chords { (0,2), (0,3), ..., (0,20) } works.
    
    # Let's build the graph and cycles explicitly.
    
    edges = set()
    
    # Base cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Chords
    chords = []
    for k in range(2, 21):  # 2 to 20 inclusive
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        chords.append(k)
        
    # Generate cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            # 0-1-2-...-39-0
            cycle_verts = list(range(N))
        else:
            # Find k
            # L = k+1 => k = L-1
            # L = N-k+1 => k = N-L+1
            # We prefer the smaller path for simplicity, but any valid cycle works.
            # If L <= 21, use k = L-1 (short path 0..k)
            # If L > 21, use k = N-L+1 (long path 0..k via 39)
            
            if L <= 21:
                k = L - 1
                # Cycle: 0, 1, ..., k, 0
                cycle_verts = list(range(k + 1))
            else:
                k = N - L + 1
                # Cycle: 0, N-1, N-2, ..., k, 0
                # Vertices: 0, 39, 38, ..., k
                # Let's trace: 0 -> 39 -> ... -> k -> 0
                # Path: 0 to k going backwards.
                # Vertices: [0] + list(range(N-1, k-1, -1))
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle_verts)
        
    # Validation
    edge_set = set(tuple(sorted(e)) for e in edges)
    valid = True
    
    # Check edges
    if len(edge_set) != len(edges):
        valid = False
    for u, v in edge_set:
        if not (0 <= u < N and 0 <= v < N and u != v):
            valid = False
            
    # Check cycles
    if len(cycles) != N - 2:
        valid = False
    else:
        for idx, L in enumerate(range(3, N + 1)):
            cycle = cycles[idx]
            if len(cycle) != L:
                valid = False
                break
            if len(set(cycle)) != L:
                valid = False
                break
            for i in range(L):
                u = cycle[i]
                v = cycle[(i + 1) % L]
                if (min(u, v), max(u, v)) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return {'edges': [tuple(sorted(e)) for e in edges], 'cycles': cycles}
    else:
        # Fallback to fan if validation fails (should not happen)
        fan_edges = set()
        for i in range(N - 1):
            fan_edges.add((i, i + 1))
        for i in range(2, N):
            fan_edges.add((0, i))
            
        fan_cycles = []
        for k in range(3, N + 1):
            fan_cycles.append(list(range(k)))
            
        return {'edges': [tuple(sorted(e)) for e in fan_edges], 'cycles': fan_cycles}
