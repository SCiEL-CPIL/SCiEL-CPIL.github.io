"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5 seconds buffer
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    random.seed(42)
    
    # Local Search / Hill Climbing
    # We try to find a set U that maximizes the score.
    # Moves: Add element, Remove element, Perturb element.
    
    def get_neighborhood(U, max_val):
        """Generate a list of neighbor sets."""
        neighbors = []
        U_set = set(U)
        
        # 1. Try adding a random element
        # Sample from a range that seems plausible based on current max
        sample_max = max(max_val, max(U) * 2 + 10)
        for _ in range(5):
            new_val = random.randint(1, sample_max)
            if new_val not in U_set:
                new_U = U_set | {new_val}
                neighbors.append(list(new_U))
        
        # 2. Try removing a random element (not 0)
        if len(U) > 2: # Keep at least 0 and one other
            non_zero = [x for x in U if x != 0]
            for _ in range(5):
                rem_val = random.choice(non_zero)
                new_U = U_set - {rem_val}
                neighbors.append(list(new_U))
        
        # 3. Try perturbing a random element
        if len(U) > 2:
            non_zero = [x for x in U if x != 0]
            for _ in range(5):
                idx = random.choice(non_zero)
                # Perturb by small amount
                delta = random.randint(-5, 5)
                new_val = idx + delta
                if new_val >= 0 and new_val != idx and new_val not in U_set:
                    new_U = U_set - {idx} | {new_val}
                    neighbors.append(list(new_U))
                    
        return neighbors

    # Initialize with several random starting points
    # Strategy: Start with sets of varying sizes and spreads
    initial_sets = []
    
    # Small dense set
    initial_sets.append([0, 1, 2, 3, 4, 5])
    # Sparse geometric
    initial_sets.append([0, 1, 2, 4, 8, 16, 32])
    # Random medium
    for _ in range(5):
        size = random.randint(10, 30)
        max_val = random.randint(50, 500)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        initial_sets.append(list(U))
        
    # Run local search on each initial set
    for init_U in initial_sets:
        current_U = sorted(init_U)
        current_score = compute_score(current_U)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_U[:]
            
        no_improvement_count = 0
        max_no_improvement = 1000 # Restart if stuck
        
        while time.time() - start_time < time_limit:
            if no_improvement_count > max_no_improvement:
                break
            
            neighbors = get_neighborhood(current_U, max(current_U) if current_U else 10)
            
            improved = False
            best_neighbor = current_U
            best_neighbor_score = current_score
            
            for n_U in neighbors:
                n_score = compute_score(n_U)
                if n_score > best_neighbor_score:
                    best_neighbor_score = n_score
                    best_neighbor = n_U
                    improved = True
            
            if improved:
                current_U = best_neighbor
                current_score = best_neighbor_score
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_U[:]
            else:
                no_improvement_count += 1
                
            # Occasionally try a larger jump (restart-like behavior within the loop)
            if no_improvement_count % 100 == 0:
                # Try a more aggressive perturbation
                if len(current_U) > 5:
                    # Remove a few elements
                    to_remove = random.sample([x for x in current_U if x != 0], min(3, len(current_U)-1))
                    new_U = [x for x in current_U if x not in to_remove]
                    # Add some new random elements
                    max_val = max(new_U) + 50 if new_U else 50
                    for _ in range(3):
                        new_U.append(random.randint(1, max_val))
                    new_score = compute_score(new_U)
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        no_improvement_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current_U[:]

    return best_set
# EVOLVE-BLOCK-END
