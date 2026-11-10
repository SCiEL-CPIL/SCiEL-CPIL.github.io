import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions
    # Store as (a, d) pairs where a is 0-based start, d is difference
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    a_vals = aps_arr[:, 0]
    d_vals = aps_arr[:, 1]
    
    # Precompute indices for each AP
    # idx1 = a, idx2 = a+d, idx3 = a+2d, idx4 = a+3d
    idx1 = a_vals
    idx2 = a_vals + d_vals
    idx3 = a_vals + 2 * d_vals
    idx4 = a_vals + 3 * d_vals

    def score(colors):
        """
        Calculate the number of monochromatic 4-APs.
        colors: numpy array of shape (N,) with values 0 or 1.
        """
        c1 = colors[idx1]
        c2 = colors[idx2]
        c3 = colors[idx3]
        c4 = colors[idx4]
        # Monochromatic if all 4 are equal
        # This is equivalent to: (c1==c2) & (c2==c3) & (c3==c4)
        # Or more efficiently using XOR: c1^c2 == 0, c2^c3 == 0, c3^c4 == 0
        mono = (c1 == c2) & (c2 == c3) & (c3 == c4)
        return int(np.sum(mono))

    def get_delta(colors, i):
        """
        Calculate the change in score if we flip bit i.
        We need to count how many APs containing i are currently monochromatic
        and how many would become monochromatic after the flip.
        
        Let S be the set of APs containing i.
        For an AP in S, let the other 3 colors be c_j, c_k, c_l.
        Current color at i is c_i.
        
        If c_i == c_j == c_k == c_l, the AP is currently mono. After flip, it is not. (-1)
        If c_i != c_j, c_i != c_k, c_i != c_l (i.e., other 3 are same, and c_i is different),
        the AP is currently NOT mono. After flip, c_i becomes same as others, so it IS mono. (+1)
        
        So delta = (count of APs where other 3 are same as c_i) * (-1) + (count of APs where other 3 are same as 1-c_i) * (+1)
        """
        c_i = colors[i]
        # Find all APs containing i
        # An AP (a, d) contains i if i is one of a, a+d, a+2d, a+3d.
        # We can iterate through all APs and check, but that's O(M).
        # Better: precompute for each index i, the list of (ap_index, position_in_ap)
        # But let's try a direct check first to see if it's fast enough, or precompute.
        
        # Precomputation strategy:
        # For each index i, store a list of tuples (other_idx1, other_idx2, other_idx3)
        # where the AP is (i, o1, o2, o3) in some order.
        # Actually, we just need the colors of the other 3 elements.
        
        # Let's use the precomputed index_to_aps structure from the thought process
        # But we haven't built it yet. Let's build it inside construct() once.
        
        # Since this function is called many times, we should pass precomputed data or use a closure.
        # Let's assume `index_to_aps` is available in the closure.
        
        delta = 0
        for (ap_idx, pos) in index_to_aps[i]:
            # Get the other 3 indices
            # ap_idx corresponds to aps_arr[ap_idx]
            # pos is 0, 1, 2, or 3
            # We can just look up the 4 indices and pick the other 3
            a, d = aps_arr[ap_idx]
            # The 4 indices are a, a+d, a+2d, a+3d
            # We know one of them is i.
            # We can just get the colors of the 4 indices and exclude the one at i.
            # Or simpler: get the 4 colors, and check the condition.
            
            c1 = colors[idx1[ap_idx]]
            c2 = colors[idx2[ap_idx]]
            c3 = colors[idx3[ap_idx]]
            c4 = colors[idx4[ap_idx]]
            
            # We need to check if the other 3 are equal to c_i or 1-c_i
            # Let's just gather the other 3
            if pos == 0:
                others = [c2, c3, c4]
            elif pos == 1:
                others = [c1, c3, c4]
            elif pos == 2:
                others = [c1, c2, c4]
            else:
                others = [c1, c2, c3]
                
            # Check if others are all equal to c_i
            if others[0] == c_i and others[1] == c_i and others[2] == c_i:
                delta -= 1
            # Check if others are all equal to 1-c_i
            elif others[0] == 1-c_i and others[1] == 1-c_i and others[2] == 1-c_i:
                delta += 1
                
        return delta

    # Build index_to_aps
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (a, d) in enumerate(aps):
        i1 = a
        i2 = a + d
        i3 = a + 2 * d
        i4 = a + 3 * d
        index_to_aps[i1].append((ap_idx, 0))
        index_to_aps[i2].append((ap_idx, 1))
        index_to_aps[i3].append((ap_idx, 2))
        index_to_aps[i4].append((ap_idx, 3))

    # Initialize with random coloring
    colors = np.random.randint(0, 2, size=N, dtype=np.int8)
    current_score = score(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9999
    min_T = 1e-5
    
    # We will do multiple passes over the bits
    # To avoid bias, we randomize the order of bits in each pass
    order = np.arange(N)
    
    while time.time() < deadline and T > min_T:
        np.random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 1.0:
                break
                
            delta = get_delta(colors, i)
            
            # Acceptance probability
            if delta < 0:
                # Improvement
                colors[i] ^= 1
                current_score += delta
            else:
                # Deterioration or neutral
                # Accept with probability exp(-delta / T)
                # If delta is 0, exp(0)=1, so always accept (random walk)
                if delta == 0:
                    if np.random.random() < 0.5: # Random walk on flat spots
                        colors[i] ^= 1
                else:
                    # delta > 0
                    if np.random.random() < np.exp(-delta / T):
                        colors[i] ^= 1
                        current_score += delta
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                
        T *= cooling
        
        # If we reach 0, we can stop or restart with a higher T
        if best_score == 0:
            break
            
    return best_colors.tolist()

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result length: {len(res)}")
    assert len(res) == 256
    assert all(c in [0, 1] for c in res)
    print("Valid")
