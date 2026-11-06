import itertools
import random
import sys

def construct():
    vertices = list(range(16))
    all_triples = list(itertools.combinations(vertices, 3))
    triple_to_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) structures
    # A structure is defined by 3 disjoint pairs.
    # We generate all sets of 6 vertices, then all partitions into 3 pairs.
    structures = [] # List of lists of 8 triples (as indices)
    triple_to_structures = [[] for _ in range(len(all_triples))]
    
    # Generate all 6-subsets
    for six_verts in itertools.combinations(vertices, 6):
        # Generate all ways to partition 6 items into 3 pairs
        # We can do this by picking the first element, pairing it with any of the other 5,
        # then picking the next smallest, pairing with any of remaining 3, etc.
        # To avoid duplicates, we enforce order on the pairs and within pairs?
        # Actually, the 8 triples are unique regardless of pair order.
        # But we must ensure we don't generate the same set of 8 triples multiple times.
        # A set of 8 triples corresponds uniquely to a partition of 6 vertices into 3 pairs.
        
        # Helper to get all pairings of 6 elements
        def get_pairings(verts):
            # verts is a tuple of 6 distinct ints
            # Return a list of 15 tuples of 3 pairs (frozensets)
            v = list(verts)
            pairs = []
            # Pair v[0] with v[1..5]
            for i in range(1, 6):
                p1 = frozenset((v[0], v[i]))
                remaining = [v[j] for j in range(6) if j != 0 and j != i]
                # Now pair remaining 4
                # Pair remaining[0] with remaining[1..3]
                for j in range(1, 4):
                    p2 = frozenset((remaining[0], remaining[j]))
                    p3 = frozenset((remaining[1], remaining[2]) if j != 1 else (remaining[2], remaining[3]))
                    # Wait, indices in remaining change.
                    # Let's just generate all 15 explicitly or use a standard algorithm.
                    pass
            
            # Standard algorithm for perfect matchings of K6
            # Fix v[0]. Pair with v[i] for i=1..5.
            # Remaining 4. Fix smallest remaining r0. Pair with rj for j in remaining indices > index of r0.
            # This generates 15 unique matchings.
            matchings = []
            v0 = v[0]
            for i in range(1, 6):
                p1 = (v0, v[i])
                rem = [v[k] for k in range(6) if k != 0 and k != i]
                # rem has 4 elements.
                # Fix rem[0]. Pair with rem[j] for j=1,2,3.
                r0 = rem[0]
                for j in range(1, 4):
                    p2 = (r0, rem[j])
                    # Last two form p3
                    # The indices of the last two in rem are the ones not 0 and not j
                    # rem is a list.
                    # Indices used: 0, j.
                    # Remaining indices: 1, 2, 3 minus j.
                    # If j=1, rem[2], rem[3]
                    # If j=2, rem[1], rem[3]
                    # If j=3, rem[1], rem[2]
                    if j == 1:
                        p3 = (rem[2], rem[3])
                    elif j == 2:
                        p3 = (rem[1], rem[3])
                    else:
                        p3 = (rem[1], rem[2])
                    
                    # Create the 8 triples
                    # Each triple picks one from each pair
                    # p1=(a,b), p2=(c,d), p3=(e,f)
                    # Triples: (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f)
                    a, b = p1
                    c, d = p2
                    e, f = p3
                    tris = [
                        (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                        (b, c, e), (b, c, f), (b, d, e), (b, d, f)
                    ]
                    # Sort each triple for consistency
                    tris = [tuple(sorted(t)) for t in tris]
                    # Convert to indices
                    tris_idx = [triple_to_index[t] for t in tris]
                    matchings.append(tuple(tris_idx))
            return matchings

        for m in get_pairings(six_verts):
            structures.append(m)
            for t_idx in m:
                triple_to_structures[t_idx].append(len(structures) - 1)

    num_structures = len(structures)
    
    best_size = 0
    best_set = set()
    
    # Run multiple restarts
    for restart in range(200):
        current_set = set()
        available = set(range(len(all_triples)))
        
        # count_in_structure[s] = number of triples from structure s in current_set
        counts = [0] * num_structures
        
        while available:
            # Find the best triple to add
            best_t = -1
            best_score = float('inf')
            
            # To speed up, we can sample or just iterate all.
            # 560 triples is small.
            for t_idx in available:
                score = 0
                safe = True
                for s_idx in triple_to_structures[t_idx]:
                    c = counts[s_idx]
                    if c == 7:
                        safe = False
                        break
                    # Penalty based on how close to limit
                    if c == 6:
                        score += 100
                    elif c == 5:
                        score += 10
                    elif c == 4:
                        score += 1
                    # else 0
                
                if not safe:
                    continue
                
                # Add small random noise to break ties and encourage diversity
                score += random.random() * 0.1
                
                if score < best_score:
                    best_score = score
                    best_t = t_idx
            
            if best_t == -1:
                # No safe triple found
                break
            
            # Add best_t
            current_set.add(best_t)
            available.discard(best_t)
            
            # Update counts
            for s_idx in triple_to_structures[best_t]:
                counts[s_idx] += 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # Early exit if we hit a theoretical upper bound?
            # Max triples is 560.
            # If we get close, we can stop.
            if best_size > 400:
                break

    return [list(all_triples[t]) for t in best_set]
