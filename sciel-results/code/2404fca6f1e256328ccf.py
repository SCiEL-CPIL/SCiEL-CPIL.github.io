"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # Precompute subset masks as numpy array
    all_masks = np.arange(total_subsets, dtype=np.int32)
    
    best_result = None
    best_score = -1
    
    # Try multiple random restarts to find a good solution
    for trial in range(30):
        random.seed(trial)
        selected_set = set()
        selected_list = []
        
        # Start with a random mask
        start = random.randint(0, total_subsets - 1)
        selected_set.add(start)
        selected_list.append(start)
        
        # Greedily add masks that maximize comparable pairs with existing selected masks
        for _ in range(m - 1):
            # Compute score for each candidate: number of comparable pairs with selected masks
            scores = np.zeros(total_subsets, dtype=np.int32)
            
            for sel in selected_list:
                # Find all masks comparable with sel
                # A mask A is comparable with sel if:
                #   (A & sel) == A and A != sel  (A is strict subset of sel)
                #   or (A & sel) == sel and A != sel  (A is strict superset of sel)
                subset_of_sel = ((all_masks & sel) == all_masks) & (all_masks != sel)
                superset_of_sel = ((all_masks & sel) == sel) & (all_masks != sel)
                comparable = subset_of_sel | superset_of_sel
                scores += comparable.astype(np.int32)
            
            # Exclude already selected masks
            for sel in selected_set:
                scores[sel] = -1
            
            # Select the mask with highest score
            best_idx = int(np.argmax(scores))
            selected_set.add(best_idx)
            selected_list.append(best_idx)
        
        # Count total comparable pairs in this solution
        total_comparable = 0
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                if ((a & b) == a and a != b) or ((a & b) == b and a != b):
                    total_comparable += 1
        
        if total_comparable > best_score:
            best_score = total_comparable
            best_result = selected_list
    
    # Convert selected masks to lists of integers
    result = []
    for mask in best_result:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
