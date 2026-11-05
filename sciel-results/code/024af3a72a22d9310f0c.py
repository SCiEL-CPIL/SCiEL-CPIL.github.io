import time
import random
import sys

N = 256
DEADLINE = 85.0

# Precompute 4-AP masks
FAP_MASKS = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            FAP_MASKS.append(mask)

# Precompute 3-AP masks
TAP_MASKS = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = b + d
        if c <= N:
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            TAP_MASKS.append(mask)

def has_4ap(mask):
    for m in FAP_MASKS:
        if (mask & m) == m:
            return True
    return False

def count_3aps(mask):
    count = 0
    for m in TAP_MASKS:
        if (mask & m) == m:
            count += 1
    return count

def greedy_initial():
    mask = 0
    for x in range(1, N + 1):
        bit = 1 << (x-1)
        new_mask = mask | bit
        if not has_4ap(new_mask):
            mask = new_mask
    return mask

def simulated_annealing(initial_mask, time_limit):
    start = time.time()
    current_mask = initial_mask
    current_score = count_3aps(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    T = 5.0
    cooling = 0.9995
    
    elements = list(range(1, N + 1))
    
    while time.time() - start < time_limit:
        # Choose move: 70% swap, 30% flip
        if random.random() < 0.7 and current_score > 0:
            # Swap
            # Pick random x not in mask
            # This is slow if done naively. 
            # We can maintain a list of unselected elements?
            # Or just try a few random ones.
            x = random.choice(elements)
            if current_mask & (1 << (x-1)):
                continue # Already in mask, skip or pick another
            
            y = random.choice(elements)
            if not (current_mask & (1 << (y-1))):
                continue # y not in mask, skip
            
            bit_x = 1 << (x-1)
            bit_y = 1 << (y-1)
            new_mask = (current_mask | bit_x) & ~bit_y
            
            if not has_4ap(new_mask):
                new_score = count_3aps(new_mask)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
        else:
            # Flip
            x = random.choice(elements)
            bit = 1 << (x-1)
            if current_mask & bit:
                new_mask = current_mask & ~bit
                new_score = count_3aps(new_mask)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                new_mask = current_mask | bit
                if not has_4ap(new_mask):
                    new_score = count_3aps(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        
        T *= cooling
        if T < 0.01:
            break
            
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    # Initial greedy
    mask = greedy_initial()
    
    # Run SA
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 1.0:
        mask, score = simulated_annealing(mask, remaining - 2.0)
    
    # Convert mask to list
    result = [i+1 for i in range(N) if (mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    result = construct()
    mask = 0
    for r in result:
        mask |= 1 << (r-1)
    print(f"Size: {len(result)}, Score: {count_3aps(mask)}, Valid: {not has_4ap(mask)}")
