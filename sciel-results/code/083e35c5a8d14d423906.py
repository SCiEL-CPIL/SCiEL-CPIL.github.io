import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    best_edges = None
    best_count = float('inf')
    best_n = None

    # Try different n values
    # n=11 is known to have a 51-edge solution
    # n=12 might have fewer edges? Unlikely, usually more vertices allow sparser graphs, 
    # but the "density" of constraints changes. 
    # The problem asks to minimize edges. 
    # Known bounds: m(5) <= 51.
    # So we aim for 51.
    
    for n in [11, 12, 13]:
        if time.time() > deadline - 5.0:
            break
            
        # Generate all 5-uniform edges
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        
        # Precompute bitsets for "Zero" and "One" monochromatic colorings
        # Bit i in the integer represents coloring i
        # There are 2^n colorings.
        num_colorings = 1 << n
        
        # Z[e] is a bitset of colorings where edge e is all 0
        # O[e] is a bitset of colorings where edge e is all 1
        Z = [0] * len(all_edges)
        O = [0] * len(all_edges)
        
        # Efficient precomputation
        # Iterate through all colorings
        for c in range(num_colorings):
            # Check each edge? 
            # Better: For each edge, we can compute the bitset directly?
            # No, iterating colorings and checking all edges is 2048 * 462 ~ 1M ops.
            # But we can do it smarter.
            pass
        
        # Let's do the standard way: for each edge, build the bitset
        # This is 462 * 2048 = ~1M ops. 
        # Python loops are slow. 1M ops is ~0.1s.
        # We have 120s. We can do this for n=11, 12, 13.
        
        for idx, e in enumerate(all_edges):
            mask = 0
            for v in e:
                mask |= (1 << v)
            
            z_bits = 0
            o_bits = 0
            
            # Iterate colorings
            # Optimization: 
            # c & mask == 0  => c has 0s at all positions in mask
            # c & mask == mask => c has 1s at all positions in mask
            
            # We can generate these sets recursively or just loop.
            # Looping 2048 times per edge is fine.
            for c in range(num_colorings):
                if (c & mask) == 0:
                    z_bits |= (1 << c)
                elif (c & mask) == mask:
                    o_bits |= (1 << c)
            
            Z[idx] = z_bits
            O[idx] = o_bits
            # S_e = Z_e | O_e
            
        # Run Greedy Set Cover with multiple restarts
        all_mask = (1 << num_colorings) - 1
        
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            covered = 0
            selected = []
            # Randomize the order of edges to break ties and find different covers
            # We will maintain a list of available edges and pick the best one
            # To speed up, we can just iterate all edges each time.
            
            # To allow random restarts, we can slightly perturb the scores or shuffle
            # But standard greedy is deterministic. 
            # We can use "Lazy" greedy or random tie-breaking.
            
            # Let's implement a simple greedy with random tie-breaking
            # Or just run greedy once, then try to swap edges?
            # Given time, multiple random shuffles of the initial edge list 
            # doesn't help if we always pick the max gain.
            # We need to pick from top-K candidates randomly.
            
            while covered != all_mask:
                max_gain = -1
                best_candidates = []
                
                for idx in range(len(all_edges)):
                    if idx in selected:
                        continue
                    s_e = Z[idx] | O[idx]
                    gain = (s_e & ~covered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [idx]
                    elif gain == max_gain:
                        best_candidates.append(idx)
                
                if max_gain == 0:
                    # No edge covers new colorings, but we haven't covered all?
                    # This shouldn't happen if the hypergraph is non-2-colorable,
                    # but we are building it. If we can't cover, this n is bad?
                    # Actually, if union of all S_e != all_mask, then there is a 
                    # coloring that makes NO edge monochromatic. 
                    # That means the complete hypergraph is 2-colorable?
                    # No, complete hypergraph is never 2-colorable for n>=10?
                    # Wait. If n=11, complete hypergraph is not 2-colorable.
                    # So union of all S_e MUST be all_mask.
                    # If we get stuck, it's a bug.
                    break
                
                # Randomly pick one of the best candidates
                chosen = random.choice(best_candidates)
                selected.append(chosen)
                covered |= (Z[chosen] | O[chosen])
                
            if covered == all_mask:
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_edges = [list(all_edges[i]) for i in selected]
                    best_n = n
                    # If we found 51, we can stop early? 
                    # 51 is the known optimum. 
                    if best_count <= 51:
                        # Try to find even better? Unlikely.
                        # But let's finish this n to see if we can get < 51?
                        # No, 51 is the lower bound for this construction family?
                        # Actually, m(5) is the minimum number of edges for ANY 
                        # 5-uniform hypergraph to be non-2-colorable.
                        # If we find 50, that would be a new record!
                        # But let's assume 51 is optimal.
                        pass

    if best_edges is None:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
