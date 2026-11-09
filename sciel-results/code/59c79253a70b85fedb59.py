import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    def get_all_block_products(seq):
        """Get all nonempty consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if sequence is valid: strictly increasing and all block products distinct."""
        if not seq:
            return True
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        # Check all block products are distinct
        products = get_all_block_products(seq)
        total_blocks = len(seq) * (len(seq) + 1) // 2
        return len(products) == total_blocks
    
    def greedy_construct():
        """Build sequence greedily, trying to add the smallest valid number at each step."""
        seq = []
        used_products = set()
        
        for val in range(2, 129):
            # Check if we can add val
            valid = True
            new_products = set()
            
            # Singleton
            if val in used_products:
                valid = False
            else:
                new_products.add(val)
            
            # Consecutive blocks ending at val
            if valid and seq:
                prod = val
                for i in range(len(seq) - 1, -1, -1):
                    prod *= seq[i]
                    if prod in used_products:
                        valid = False
                        break
                    new_products.add(prod)
            
            if valid:
                seq.append(val)
                used_products.update(new_products)
        
        return seq
    
    def local_search(seq, max_iter=500):
        """Improve sequence via local search: try removing and reinserting elements."""
        current = seq[:]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new element
            for val in range(2, 129):
                if val in current:
                    continue
                # Find position to insert (maintaining sorted order)
                pos = 0
                while pos < len(current) and current[pos] < val:
                    pos += 1
                
                new_seq = current[:pos] + [val] + current[pos:]
                if is_valid(new_seq):
                    current = new_seq
                    improved = True
                    break
            
            # Try removing and reinserting an element
            if not improved and len(current) > 0:
                for i in range(len(current)):
                    val = current[i]
                    # Remove val
                    temp_seq = current[:i] + current[i+1:]
                    
                    # Try reinserting at different positions
                    for j in range(len(temp_seq) + 1):
                        if j == i:
                            continue
                        new_seq = temp_seq[:j] + [val] + temp_seq[j:]
                        if is_valid(new_seq) and len(new_seq) >= len(current):
                            current = new_seq
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current
    
    def simulated_annealing(seq, max_iter=2000):
        """Use simulated annealing to find longer valid sequences."""
        current = seq[:]
        best = current[:]
        best_len = len(current)
        
        temperature = 5.0
        cooling = 0.998
        
        for iter_num in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate a neighbor
            neighbor = current[:]
            action = random.random()
            
            if action < 0.3:
                # Remove a random element
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor) - 1)
                    neighbor.pop(idx)
            elif action < 0.6:
                # Swap two elements
                if len(neighbor) >= 2:
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    # Ensure strictly increasing
                    if not all(neighbor[k] < neighbor[k+1] for k in range(len(neighbor)-1)):
                        neighbor = current[:]  # Revert
            else:
                # Replace a random element with another value
                if len(neighbor) > 0:
                    idx = random.randint(0, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    new_val = random.randint(2, 128)
                    if new_val != old_val:
                        neighbor[idx] = new_val
                        # Ensure strictly increasing
                        if not all(neighbor[k] < neighbor[k+1] for k in range(len(neighbor)-1)):
                            neighbor = current[:]  # Revert
            
            # Check validity
            if not is_valid(neighbor):
                continue
            
            neighbor_len = len(neighbor)
            
            # Accept if better, or with probability based on temperature
            if neighbor_len > len(current):
                current = neighbor
                if neighbor_len > best_len:
                    best = neighbor[:]
                    best_len = neighbor_len
            else:
                # Compute energy difference (negative is better)
                energy_diff = len(current) - neighbor_len
                if random.random() < pow(2.718281828, -energy_diff / temperature):
                    current = neighbor
            
            temperature *= cooling
        
        return best
    
    # Run multiple strategies
    best_seq = []
    best_len = 0
    
    # Strategy 1: Greedy
    greedy_result = greedy_construct()
    if len(greedy_result) > best_len:
        best_seq = greedy_result
        best_len = len(greedy_result)
    
    # Strategy 2: Local search from greedy
    if len(greedy_result) > 0:
        local_result = local_search(greedy_result, max_iter=200)
        if len(local_result) > best_len:
            best_seq = local_result
            best_len = len(local_result)
    
    # Strategy 3: Simulated annealing from greedy
    if len(greedy_result) > 0:
        sa_result = simulated_annealing(greedy_result, max_iter=2000)
        if len(sa_result) > best_len:
            best_seq = sa_result
            best_len = len(sa_result)
    
    # Strategy 4: Multiple random restarts
    random.seed(42)
    for restart in range(15):
        if time.time() > deadline:
            break
        
        # Random initial sequence
        initial_seq = []
        used = set()
        for _ in range(random.randint(5, 12)):
            val = random.randint(2, 128)
            if val not in used:
                used.add(val)
                initial_seq.append(val)
        initial_seq.sort()
        
        # Local search
        ls_result = local_search(initial_seq, max_iter=100)
        if len(ls_result) > best_len:
            best_seq = ls_result
            best_len = len(ls_result)
        
        # Simulated annealing
        sa_result = simulated_annealing(initial_seq, max_iter=500)
        if len(sa_result) > best_len:
            best_seq = sa_result
            best_len = len(sa_result)
    
    # Final validation
    if not is_valid(best_seq):
        # Fallback to a simple valid sequence
        best_seq = [2]
    
    return best_seq

# EVOLVE-BLOCK-END
