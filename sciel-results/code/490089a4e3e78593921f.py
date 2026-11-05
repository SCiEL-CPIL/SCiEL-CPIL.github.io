import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= n
    # Using 0-based indexing: 0 <= a and a+3d < n
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each bit position, the list of (ap_index, other_3_indices)
    # This allows O(degree) delta calculation
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            others = [x for x in ap if x != pos]
            bit_aps[pos].append((idx, others))
    
    def count_mono(colors):
        """Count monochromatic 4-APs. Used for initial check and final verification."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count

    def local_search(colors):
        """
        Perform local search (hill climbing) by flipping single bits.
        Returns the improved colors and the final count.
        """
        current_count = count_mono(colors)
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias, or iterate sequentially. 
            # Sequential is faster but can be order-dependent. 
            # Given small N, we can do a few passes or randomize.
            # Let's iterate 0 to n-1. If a flip improves, do it.
            for i in range(n):
                # Calculate delta if we flip colors[i]
                delta = 0
                for ap_idx, others in bit_aps[i]:
                    # Get colors of the other 3
                    c0 = colors[others[0]]
                    c1 = colors[others[1]]
                    c2 = colors[others[2]]
                    curr_c = colors[i]
                    
                    # Check if the AP was mono before
                    # Mono if all 4 same. Since we are flipping i, 
                    # it was mono iff c0==c1==c2==curr_c
                    if c0 == c1 == c2 == curr_c:
                        # Was mono (1), becomes non-mono (0)
                        delta -= 1
                    else:
                        # Was non-mono (0). 
                        # Becomes mono iff c0==c1==c2 == (1 - curr_c)
                        if c0 == c1 == c2 == (1 - curr_c):
                            # Becomes mono (1)
                            delta += 1
                        # Else remains non-mono (0)
                
                if delta < 0:
                    # Flip
                    colors[i] = 1 - colors[i]
                    current_count += delta
                    improved = True
        
        return colors, current_count

    best_colors = None
    best_count = float('inf')
    
    # Strategy 1: Thue-Morse based initialization
    # Thue-Morse sequence: t(n) = popcount(n) % 2
    # This sequence is known to be "balanced" and has low AP counts.
    tm_colors = [bin(i).count('1') % 2 for i in range(n)]
    tm_colors, tm_count = local_search(tm_colors)
    if tm_count < best_count:
        best_count = tm_count
        best_colors = tm_colors.copy()
        
    # Strategy 2: Random restarts with local search
    # We can try a few random starts or perturbed Thue-Morse.
    # Given time budget, we can afford several restarts.
    # Estimate time per local search: 
    # N=256, avg degree ~ 10-20. One pass is ~ 256*20 = 5120 ops.
    # Convergence might take 10-50 passes. ~ 50k-100k ops per search.
    # Python overhead makes it slower, but 100 searches should be feasible in < 10s.
    
    random.seed(42)
    num_restarts = 50 # Adjust based on time if needed
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random starting coloring
        # Or perturb the best found so far? 
        # Pure random is good for diversity.
        current_colors = [random.randint(0, 1) for _ in range(n)]
        
        # Run local search
        current_colors, current_count = local_search(current_colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = current_colors.copy()
            
    # If we have time, try more sophisticated perturbations of the best solution
    # e.g., flip k random bits and re-optimize
    if time.time() < deadline - 10 and best_colors:
        for r in range(20):
            if time.time() > deadline:
                break
            perturbed = best_colors.copy()
            # Flip 5 random bits
            for _ in range(5):
                idx = random.randint(0, n-1)
                perturbed[idx] = 1 - perturbed[idx]
            
            perturbed, count = local_search(perturbed)
            if count < best_count:
                best_count = count
                best_colors = perturbed.copy()

    # Final verification
    if best_colors is None:
        # Fallback to simple alternating if something went wrong
        best_colors = [(i // 2) % 2 for i in range(n)]
        best_count = count_mono(best_colors)
    else:
        # Ensure the count matches the actual count (safety check)
        actual_count = count_mono(best_colors)
        if actual_count < best_count:
            best_count = actual_count

    return best_colors
