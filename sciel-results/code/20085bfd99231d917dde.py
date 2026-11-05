"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Candidates: numbers in [1, 255] sharing a factor with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Initialize with the largest single-prime multiple set: multiples of 3
    # This is a valid clique of size 85.
    current_clique = set([i for i in candidates if i % 3 == 0])
    
    # Local search: try to add pairs of elements from outside the clique
    # by removing conflicting elements from inside, if it increases size.
    # We can also try single additions (though unlikely to help if starting from max single prime).
    
    improved = True
    while improved:
        improved = False
        outside = [c for c in candidates if c not in current_clique]
        
        # Try adding pairs (x, y) from outside
        # To be efficient, we can limit this or just run it once.
        # Given small N, a full search for pairs is feasible.
        for i in range(len(outside)):
            x = outside[i]
            for j in range(i + 1, len(outside)):
                y = outside[j]
                
                # Check if x and y are compatible
                if math.gcd(x, y) == 1:
                    continue
                    
                # Find conflicts in current_clique
                # Conflict for x: z in current_clique s.t. gcd(x,z)==1
                # Conflict for y: z in current_clique s.t. gcd(y,z)==1
                # Union of conflicts must be removed.
                conflicts = set()
                for z in current_clique:
                    if math.gcd(x, z) == 1 or math.gcd(y, z) == 1:
                        conflicts.add(z)
                        
                # New size = current_size - |conflicts| + 2
                new_size = len(current_clique) - len(conflicts) + 2
                if new_size > len(current_clique):
                    # Verify the new set is a clique
                    # x and y are compatible with each other (checked)
                    # x and y are compatible with current_clique \ conflicts (by definition of conflicts)
                    # current_clique \ conflicts is a clique (subset of original clique)
                    # So the new set is a clique.
                    current_clique = (current_clique - conflicts) | {x, y}
                    improved = True
                    # Restart loop as outside set changed
                    break
            if improved:
                break
                
    # Also try single additions just in case (though logic says no gain)
    # And try starting from other bases?
    # The local search should find the global optimum if it's reachable by swaps.
    
    return list(current_clique)
# EVOLVE-BLOCK-END
