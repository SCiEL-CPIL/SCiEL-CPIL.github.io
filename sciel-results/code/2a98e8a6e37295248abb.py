import time
import random
import itertools

# Constants
N = 256
DEADLINE = 90.0  # seconds

# Precompute 3-APs and 4-APs
# A 3-AP is (a, a+d, a+2d) with a+2d <= N
# A 4-AP is (a, a+d, a+2d, a+3d) with a+3d <= N

three_aps = []
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        three_aps.append((a, a + d, a + 2 * d))

four_aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

# For fast lookup of 4-APs containing a specific element
# Map element -> list of indices of 4-APs containing it
elem_to_4ap_idx = {i: [] for i in range(1, N + 1)}
for idx, ap in enumerate(four_aps):
    for x in ap:
        elem_to_4ap_idx[x].append(idx)

# For fast lookup of 3-APs containing a specific element
# Map element -> list of indices of 3-APs containing it
elem_to_3ap_idx = {i: [] for i in range(1, N + 1)}
for idx, ap in enumerate(three_aps):
    for x in ap:
        elem_to_3ap_idx[x].append(idx)


def count_3aps(set_bits):
    """Count number of 3-APs fully contained in the set represented by set_bits (bit i is 1 if i in set)."""
    count = 0
    for a, b, c in three_aps:
        if (set_bits & (1 << (a - 1))) and (set_bits & (1 << (b - 1))) and (set_bits & (1 << (c - 1))):
            count += 1
    return count


def has_4ap(set_bits):
    """Check if there is any 4-AP fully contained in the set represented by set_bits."""
    for a, b, c, d in four_aps:
        if (set_bits & (1 << (a - 1))) and (set_bits & (1 << (b - 1))) and (set_bits & (1 << (c - 1))) and (set_bits & (1 << (d - 1))):
            return True
    return False


def greedy_construct():
    """Construct a 4-AP-free set using a greedy strategy."""
    selected = set()
    # Try adding elements in random order or specific order
    # A common heuristic is to pick elements that don't form 4-APs with existing ones
    # To maximize 3-APs, we might want to be denser, but we must avoid 4-APs.
    # Let's use a randomized greedy approach with multiple restarts.
    
    best_set = set()
    best_score = -1
    
    # Simple greedy: iterate 1..N, add if no 4-AP formed
    # This is the seed logic, which is weak.
    # Better: Randomized greedy or heuristic based on "potential" 3-APs.
    
    # Let's just use the seed's greedy as a baseline start for local search, 
    # but also try a few randomized variants.
    
    for _ in range(5):
        s = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            # Check if adding x creates a 4-AP
            creates_4ap = False
            for idx in elem_to_4ap_idx[x]:
                ap = four_aps[idx]
                # Check if other 3 elements are in s
                others = [y for y in ap if y != x]
                if all(y in s for y in others):
                    creates_4ap = True
                    break
            if not creates_4ap:
                s.add(x)
        
        if len(s) > len(best_set):
            best_set = s
    
    # Fallback to deterministic greedy if randomized is bad?
    # Actually, let's just use the deterministic one as one of the starts.
    s_det = set()
    for x in range(1, N + 1):
        creates_4ap = False
        for idx in elem_to_4ap_idx[x]:
            ap = four_aps[idx]
            others = [y for y in ap if y != x]
            if all(y in s_det for y in others):
                creates_4ap = True
                break
        if not creates_4ap:
            s_det.add(x)
    
    if len(s_det) > len(best_set):
        best_set = s_det
        
    return best_set


def local_search(initial_set, time_limit):
    """Perform 1-swap and 2-swap local search to maximize 3-APs while keeping 4-AP free."""
    current_set = set(initial_set)
    current_bits = 0
    for x in current_set:
        current_bits |= (1 << (x - 1))
    
    current_score = count_3aps(current_bits)
    best_set = set(current_set)
    best_bits = current_bits
    best_score = current_score
    
    start_time = time.time()
    
    # Precompute neighbors for 1-swap: for each x in set, what are the y not in set?
    # This is O(N^2) per iteration if done naively.
    # We can optimize by precomputing which 4-APs are "close" to being formed.
    
    # To speed up, we can maintain a count of "potential" 4-APs.
    # But for N=256, simple checks might be fast enough if we limit iterations.
    
    # Let's use a loop that tries random swaps.
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() - start_time < time_limit:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        
        # 1. Try 1-swaps
        # Pick a random element in the set
        if current_set:
            x_in = random.choice(list(current_set))
            # Pick a random element not in the set
            # To make this faster, we can maintain a list of elements not in set
            # But for now, let's just pick a random candidate from 1..N
            # and check if it's in the set.
            
            # Better: maintain a list of outside elements
            # outside_elements = [i for i in range(1, N+1) if i not in current_set]
            # This is O(N) to build, which is fine.
            
            outside_elements = [i for i in range(1, N + 1) if i not in current_set]
            if outside_elements:
                y_out = random.choice(outside_elements)
                
                # Try swap x_in -> y_out
                # Check if new set has 4-AP
                # Remove x_in, add y_out
                new_bits = current_bits
                new_bits &= ~(1 << (x_in - 1))
                new_bits |= (1 << (y_out - 1))
                
                if not has_4ap(new_bits):
                    new_score = count_3aps(new_bits)
                    if new_score > current_score:
                        current_set.remove(x_in)
                        current_set.add(y_out)
                        current_bits = new_bits
                        current_score = new_score
                        improved = True
                        no_improvement_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                            best_bits = current_bits
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1

        # 2. Try 2-swaps
        # Pick two elements in the set, two outside
        if len(current_set) >= 2:
            outside_elements = [i for i in range(1, N + 1) if i not in current_set]
            if len(outside_elements) >= 2:
                # Randomly pick 2 from in, 2 from out
                in_list = list(current_set)
                x1, x2 = random.sample(in_list, 2)
                y1, y2 = random.sample(outside_elements, 2)
                
                new_bits = current_bits
                new_bits &= ~(1 << (x1 - 1))
                new_bits &= ~(1 << (x2 - 1))
                new_bits |= (1 << (y1 - 1))
                new_bits |= (1 << (y2 - 1))
                
                if not has_4ap(new_bits):
                    new_score = count_3aps(new_bits)
                    if new_score > current_score:
                        current_set.remove(x1)
                        current_set.remove(x2)
                        current_set.add(y1)
                        current_set.add(y2)
                        current_bits = new_bits
                        current_score = new_score
                        improved = True
                        no_improvement_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                            best_bits = current_bits
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
            
        if not improved:
             # Occasionally restart or perturb?
             # For now, just keep searching with random swaps.
             pass

    return best_set, best_score


def construct():
    start_time = time.time()
    
    # Get initial solution
    init_set = greedy_construct()
    
    # Run local search with remaining time
    # Leave 5s for cleanup/return
    time_limit = max(1.0, DEADLINE - (time.time() - start_time) - 5.0)
    
    best_set, best_score = local_search(init_set, time_limit)
    
    # Verify validity one last time
    bits = 0
    for x in best_set:
        bits |= (1 << (x - 1))
        
    if has_4ap(bits):
        # Fallback to a known valid set (e.g. greedy)
        # This should not happen if local search is correct
        return sorted(greedy_construct())
        
    return sorted(best_set)

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found set of size {len(res)}")
    # Count 3-APs
    bits = 0
    for x in res:
        bits |= (1 << (x - 1))
    print(f"3-APs: {count_3aps(bits)}")
    print(f"4-APs: {has_4ap(bits)}")
