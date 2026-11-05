import time
import random
import math

def count_mono_ap(colors):
    """Count monochromatic 4-term arithmetic progressions."""
    n = len(colors)
    count = 0
    # d max is (n-1)//3
    for d in range(1, (n - 1) // 3 + 1):
        # a max is n - 3*d
        for a in range(n - 3 * d):
            if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return [bin(i).count('1') % 2 for i in range(n)]

def get_aps_for_index(i, n):
    """
    Yield all 4-term APs (as tuples of indices) that include index i.
    An AP is (a, a+d, a+2d, a+3d).
    """
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        # Case 1: i is a
        if i + 3 * d < n:
            yield (i, i + d, i + 2 * d, i + 3 * d)
        # Case 2: i is a+d
        if i - d >= 0 and i + 2 * d < n:
            yield (i - d, i, i + d, i + 2 * d)
        # Case 3: i is a+2d
        if i - 2 * d >= 0 and i + d < n:
            yield (i - 2 * d, i - d, i, i + d)
        # Case 4: i is a+3d
        if i - 3 * d >= 0:
            yield (i - 3 * d, i - 2 * d, i - d, i)

def construct():
    n = 256
    # Start with Thue-Morse
    colors = thue_morse(n)
    current_score = count_mono_ap(colors)
    
    max_time = 110  # seconds, leaving buffer
    start_time = time.time()
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() - start_time < max_time and T > min_T:
        # Randomly select an index to flip
        i = random.randint(0, n - 1)
        
        # Calculate change in score if we flip i
        delta = 0
        for ap in get_aps_for_index(i, n):
            idxs = ap
            c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
            is_mono_before = (c0 == c1 == c2 == c3)
            
            new_ci = 1 - colors[i]
            
            # Reconstruct colors with new_ci at position i
            # We can avoid list creation for speed
            # But for clarity and n=256, it's fine.
            # Let's optimize slightly by checking directly
            if idxs[0] == i:
                new_c0 = new_ci
                new_c1 = c1
                new_c2 = c2
                new_c3 = c3
            elif idxs[1] == i:
                new_c0 = c0
                new_c1 = new_ci
                new_c2 = c2
                new_c3 = c3
            elif idxs[2] == i:
                new_c0 = c0
                new_c1 = c1
                new_c2 = new_ci
                new_c3 = c3
            else: # idxs[3] == i
                new_c0 = c0
                new_c1 = c1
                new_c2 = c2
                new_c3 = new_ci
            
            is_mono_after = (new_c0 == new_c1 == new_c2 == new_c3)
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
        
        # Accept move
        if delta < 0:
            colors[i] ^= 1
            current_score += delta
        else:
            if delta == 0:
                if random.random() < 0.5:
                    colors[i] ^= 1
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    colors[i] ^= 1
                    current_score += delta
        
        T *= cooling_rate
        
    return colors
