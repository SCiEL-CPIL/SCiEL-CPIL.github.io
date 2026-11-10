import random

N = 64

def construct():
    best_set = []
    
    # Use a fixed seed for reproducibility
    random.seed(42)
    
    # Try multiple random orders to find a large set
    for _ in range(100):
        points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(points)
        
        current_set = []
        # dist_sets[i] is a set of squared distances from current_set[i] to all other points in current_set
        dist_sets = []
        
        for p in points:
            px, py = p
            valid = True
            d_p_to_s = []
            
            # Calculate distances from P to all existing points
            for i in range(len(current_set)):
                ax, ay = current_set[i]
                d2 = (px - ax)**2 + (py - ay)**2
                d_p_to_s.append(d2)
                
            k = len(current_set)
            
            if k > 0:
                # Check if any distance from P to S is duplicated (isosceles at P)
                if len(set(d_p_to_s)) < k:
                    valid = False
                else:
                    # Check if any distance from P to A_i equals distance from A_i to A_j
                    for i in range(k):
                        if d_p_to_s[i] in dist_sets[i]:
                            valid = False
                            break
            
            if valid:
                current_set.append(p)
                
                # Update dist_sets
                # New point's dist set
                new_dist_set = set(d_p_to_s)
                dist_sets.append(new_dist_set)
                
                # Update existing points' dist sets
                for i in range(k):
                    dist_sets[i].add(d_p_to_s[i])
                    
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
