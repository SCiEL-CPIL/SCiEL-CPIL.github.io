import random
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base that are > 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort for consistency
    divisors.sort()
    
    best_score = base
    best_classes = []
    
    # Number of trials
    num_trials = 200 
    
    for trial in range(num_trials):
        random.seed(trial * 7919 + 13)
        
        # cover_count[x] = number of classes covering residue x
        cover_count = np.zeros(base, dtype=np.int32)
        
        # Initialize residues
        # Strategy: Random initialization or Greedy initialization
        # Let's try Greedy Initialization with Random Order
        r_values = {}
        m_list = divisors[:]
        random.shuffle(m_list)
        
        # Greedy Init
        for m in m_list:
            # Find r that maximizes uncovered count in that class
            # Reshape cover_count to (-1, m)
            # Note: cover_count is 0 everywhere initially, so any r is equally good.
            # But after first few, it matters.
            reshaped = cover_count.reshape(-1, m)
            # We want to maximize the number of 0s in each column
            zeros_count = (reshaped == 0).sum(axis=0)
            best_r = int(np.argmax(zeros_count))
            
            r_values[m] = best_r
            # Update cover_count
            cover_count[best_r::m] += 1
            
        # Local Search (Coordinate Descent)
        improved = True
        max_iter = 10
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            # Shuffle order to avoid bias
            m_order = divisors[:]
            random.shuffle(m_order)
            
            for m in m_order:
                current_r = r_values[m]
                
                # Calculate best r
                reshaped = cover_count.reshape(-1, m)
                zeros_count = (reshaped == 0).sum(axis=0)
                best_r = int(np.argmax(zeros_count))
                
                if best_r != current_r:
                    # Check if it actually improves
                    # We can just apply it, as argmax of zeros is optimal for this step
                    # But we must update cover_count correctly
                    
                    # Remove current r
                    cover_count[current_r::m] -= 1
                    # Add new r
                    cover_count[best_r::m] += 1
                    
                    r_values[m] = best_r
                    improved = True
        
        # Calculate final score
        uncovered = (cover_count == 0).sum()
        if uncovered < best_score:
            best_score = int(uncovered)
            best_classes = [[m, r_values[m]] for m in divisors]
            
    return best_classes
