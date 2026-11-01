import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 120  # Safety margin from 150s limit
    
    # Initial best solution
    best_U = [0, 1, 3, 7]
    best_score = calculate_score(best_U)
    
    # Parameters for search
    max_val = 100  # Search within 0..100
    current_U = set(best_U)
    current_score = best_score
    
    # Simulated Annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    
    def calculate_score(U):
        if not U or 0 not in U:
            return 0.0
        U_list = list(U)
        if not U_list:
            return 0.0
        max_u = max(U_list)
        if max_u == 0:
            return 1.0
        
        # Calculate sumset and difference set
        # Using sets for unique values
        sumset = set()
        diffset = set()
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            # Score is 1 + log(ratio)/log(...)
            # If ratio < 1, log is negative, score < 1.
            # But we want to maximize.
            pass
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        score = 1.0 + math.log(ratio) / denom
        return score

    # Main search loop
    iteration = 0
    while time.time() - start_time < timeout:
        # Generate a neighbor
        # Randomly add or remove an element
        if random.random() < 0.5:
            # Add
            candidate = random.randint(1, max_val)
            if candidate not in current_U:
                new_U = current_U | {candidate}
            else:
                new_U = current_U
        else:
            # Remove (but keep 0)
            removable = [x for x in current_U if x != 0]
            if removable:
                candidate = random.choice(removable)
                new_U = current_U - {candidate}
            else:
                new_U = current_U
        
        # Ensure 0 is in new_U
        if 0 not in new_U:
            new_U.add(0)
            
        # Calculate score
        new_score = calculate_score(new_U)
        
        # Acceptance criterion
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(new_U)
        else:
            # Simulated annealing acceptance
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if accept:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(new_U)
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        iteration += 1
        
    return best_U

# Keep the original signature
def construct():
    start_time = time.time()
    timeout = 120
    
    best_U = [0, 1, 3, 7]
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U_list = list(U)
        max_u = max(U_list)
        if max_u == 0:
            return 1.0
        
        sumset = set()
        diffset = set()
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
        
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        denom = math.log(2 * max_u + 1)
        
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_score = calc_score(best_U)
    current_U = set(best_U)
    current_score = best_score
    
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    max_val = 100
    
    while time.time() - start_time < timeout:
        if random.random() < 0.5:
            candidate = random.randint(1, max_val)
            new_U = current_U | {candidate}
        else:
            removable = [x for x in current_U if x != 0]
            if removable:
                candidate = random.choice(removable)
                new_U = current_U - {candidate}
            else:
                new_U = current_U
        
        if 0 not in new_U:
            new_U.add(0)
            
        new_score = calc_score(new_U)
        
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(new_U)
        else:
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if accept:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(new_U)
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_U
