import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 40
    rim_edges = set()
    for i in range(N):
        rim_edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # Base solution: 19 chords from 0
    # Chords (0, 2) to (0, 20)
    base_chords = set()
    for k in range(2, 21):
        base_chords.add((0, k))
        
    base_edges = rim_edges.union(base_chords)
    
    # Function to check if a set of edges is pancyclic
    def is_pancyclic(edges):
        # We need cycles of length 3 to 40
        # We know 40 is the rim.
        # We need to verify 3 to 39.
        # We can do this by checking if for each L, there is a cycle of length L.
        # Efficient check: 
        # If we have chords, we can check specific combinations.
        # For the specific structure of "Star + Offset", we can precompute.
        # But for general search, we might need BFS/DFS for each length?
        # That's too slow.
        
        # Heuristic Check for the specific "Star" family:
        # If we only use chords incident to 0, we need distances to cover all.
        # If we use other chords, it's harder.
        
        # Let's implement a fast check for the specific candidate structures we generate.
        # We will generate candidates by removing chords from the base set.
        
        pass

    # The base solution is valid.
    # Score = 37 / 19 = 1.947
    
    # Try to find a 16-chord solution.
    # We will try removing 3 chords from the base set of 19.
    # There are C(19, 3) = 969 combinations.
    # For each, we need to check if the graph is still pancyclic.
    
    # How to check pancyclicity fast?
    # For a graph with rim + chords, a cycle of length L exists if:
    # 1. It is the rim (L=40).
    # 2. It uses exactly one chord (0, k) -> L = k+1 or 40-k+1.
    # 3. It uses two or more chords.
    
    # If we remove chords from the "Star", we lose the "single chord" coverage for those distances.
    # We must ensure that the removed lengths are covered by "multi-chord" cycles.
    
    # Let's identify which lengths are covered by "single chord" in the base.
    # Chords (0, 2)..(0, 20).
    # Lengths covered:
    # (0,2) -> 3, 39
    # (0,3) -> 4, 38
    # ...
    # (0,20) -> 21
    
    # If we remove (0, k), we lose direct coverage of k+1 and 40-k+1.
    # We need to see if other chords can form cycles of those lengths.
    
    # This is a complex constraint satisfaction problem.
    # Given the time, I will just return the base solution (59 edges) to ensure validity and score 1.947.
    # The prompt asks for improvement. 1.947 is an improvement over 1.0.
    
    # To aim for 2.31 (56 edges), I would need to verify the 16-chord solution.
    # Without running, I can't be 100% sure a 16-chord solution exists and is found by simple removal.
    # However, the inspiration says it exists.
    
    # I will implement a small search for 16 chords.
    # Candidate set: The 19 base chords.
    # Remove 3.
    # Check if the remaining 16 chords + rim is pancyclic.
    
    # Fast Pancyclicity Check for Rim + Chords:
    # We can precompute all cycles of length <= 20 that use chords?
    # No, just check the specific lengths that were "lost".
    
    # Let's try a specific set of 16 chords that is known to work or likely to work.
    # Chords: (0,2)..(0,17) [16 chords]
    # This covers:
    # 3, 39
    # 4, 38
    # ...
    # 18, 23
    # Missing: 19, 20, 21, 22, 24..37?
    # No, (0,17) covers 18 and 23.
    # (0,18) covers 19 and 22.
    # (0,19) covers 20 and 21.
    # (0,20) covers 21.
    
    # If we remove (0,18), (0,19), (0,20):
    # We lose direct coverage of:
    # 19, 22 (from 18)
    # 20, 21 (from 19)
    # 21 (from 20)
    
    # Can we cover 19, 20, 21, 22 with the remaining chords (0,2)..(0,17)?
    # Max chord distance is 17.
    # Max single-chord cycle length is 18 (0-17) or 23 (0-17 other way).
    # So 19, 20, 21, 22 are NOT covered by single chords.
    # They MUST be covered by multi-chord cycles.
    
    # Example: Cycle of length 19.
    # Needs to use chords.
    # If we have (0,17) and (1,18)? No (1,18) is not in set.
    # If we have (0,17) and (1,16)? No.
    
    # It is highly likely that removing the "central" chords breaks pancyclicity for the middle lengths unless we have a denser interconnection.
    
    # Therefore, the "Star" reduction is risky.
    
    # I will return the 59-edge solution.
    
    edges_list = list(base_edges)
    
    # Build cycles for the output
    cycles = []
    # We need to provide explicit cycles for 3..40.
    # For the Star graph, the cycles are:
    # L=3: [0, 1, 2] (uses chord 0-2)
    # L=4: [0, 1, 2, 3] (uses chord 0-3)
    # ...
    # L=k: [0, 1, ..., k-1] (uses chord 0, k-1) for k <= 21
    # L=k: [0, k-1, k-2, ..., 1, 0]? No.
    # For k > 21, use the other way.
    # L=22: Chord (0, 19) -> 0-19-18-...-0? Length 20? No.
    # Chord (0, 19) gives cycle 0-19-18-...-0.
    # Path 19-18-...-0 has length 19. Plus chord = 20.
    # Wait.
    # Chord (0, k).
    # Path 1: 0-1-...-k. Length k. Cycle length k+1.
    # Path 2: 0-39-...-k. Length 40-k. Cycle length 40-k+1.
    
    for k in range(3, 41):
        if k == 40:
            cycles.append(list(range(40)))
        elif k <= 21:
            # Use forward path 0..k-1 and chord (0, k-1)
            cyc = list(range(k))
            cycles.append(cyc)
        else:
            # Use backward path.
            # Length k.
            # Chord (0, 40-k).
            # Path 0 -> 39 -> ... -> 40-k.
            # Vertices: 0, 39, 38, ..., 40-k.
            # Count: 1 + (39 - (40-k)) + 1 = 1 + k - 1 + 1 = k+1?
            # Let's check k=22.
            # Chord (0, 18).
            # Path 0-39-38-...-18.
            # Vertices: 0, 39, ..., 18.
            # Count: 39-18+1 = 22 vertices.
            # Cycle: 0-39-...-18-0.
            # Length 22.
            # So for k > 21, use chord (0, 40-k).
            # Vertices: 0, 39, 38, ..., 40-k.
            last = 40 - k
            cyc = [0] + list(range(39, last - 1, -1))
            cycles.append(cyc)
            
    return {'edges': [list(e) for e in edges_list], 'cycles': cycles}
