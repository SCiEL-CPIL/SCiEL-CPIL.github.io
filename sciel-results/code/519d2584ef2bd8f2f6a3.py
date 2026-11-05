import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    best_set = []
    best_size = 0
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def is_valid_swap(A, in_A, sum_counts, y, x):
        """Check if swapping y with x is valid without modifying state."""
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        # Count how many times y contributes to sum s
        # Pairs in A summing to s involving y:
        # 1. (y, y) if 2y == s
        # 2. (y, z) if y + z == s => z = s - y. If z in A and z != y.
        contrib_y = 0
        if 2 * y == s:
            contrib_y = 1
        else:
            z = s - y
            if z in in_A and z != y:
                contrib_y = 1
        
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 0
            if 2 * y == s:
                contrib_y = 1
            else:
                w = s - y
                if w in in_A and w != y:
                    contrib_y = 1
            
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
        return True

    def perform_swap(A, in_A, sum_counts, y, x):
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1

    def is_valid_1_for_2(A, in_A, sum_counts, y, x1, x2):
        """Check if removing y and adding x1, x2 is valid."""
        if y not in in_A or x1 in in_A or x2 in in_A or x1 == x2:
            return False
        
        # Helper to get count in A' for a sum s
        # count_A'(s) = count_A(s) - count_y(s) + count_x1(s) + count_x2(s)
        
        def count_y_contribution(s):
            if 2*y == s:
                return 1
            z = s - y
            if z in in_A and z != y:
                return 1
            return 0

        def count_x_contribution(s, x):
            if 2*x == s:
                return 1
            z = s - x
            if z in in_A and z != y and z != x:
                return 1
            return 0

        # Check 2*x1
        s = 2*x1
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check 2*x2
        s = 2*x2
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check x1+x2
        s = x1 + x2
        c = sum_counts[s] - count_y_contribution(s) + count_x_contribution(s, x1) + count_x_contribution(s, x2)
        if c > 2:
            return False

        # Check x1+z and x2+z for z in A \ {y}
        for z in A:
            if z == y:
                continue
            
            # Check x1 + z
            s = x1 + z
            contrib_y = count_y_contribution(s)
            # contrib_x1 is 1 because (x1, z) is a pair in A'
            # contrib_x2: is there a pair (x2, w) in A' summing to s?
            # w = s - x2 = x1 + z - x2
            contrib_x2 = 0
            if 2*x2 == s:
                contrib_x2 = 1
            else:
                w = s - x2
                if w in in_A and w != y and w != x2:
                    contrib_x2 = 1
            
            new_count = sum_counts[s] - contrib_y + 1 + contrib_x2
            if new_count > 2:
                return False
                
            # Check x2 + z
            s = x2 + z
            contrib_y = count_y_contribution(s)
            # contrib_x2 is 1 because (x2, z) is a pair in A'
            # contrib_x1: is there a pair (x1, w) in A' summing to s?
            # w = s - x1 = x2 + z - x1
            contrib_x1 = 0
            if 2*x1 == s:
                contrib_x1 = 1
            else:
                w = s - x1
                if w in in_A and w != y and w != x1:
                    contrib_x1 = 1
            
            new_count = sum_counts[s] - contrib_y + 1 + contrib_x1
            if new_count > 2:
                return False
                
        return True

    def perform_1_for_2(A, in_A, sum_counts, y, x1, x2):
        remove_element(A, in_A, sum_counts, y)
        # Add x1
        in_A.add(x1)
        A.append(x1)
        sum_counts[2*x1] += 1
        for z in A[:-1]:
            sum_counts[x1+z] += 1
        # Add x2
        in_A.add(x2)
        A.append(x2)
        sum_counts[2*x2] += 1
        for z in A[:-1]:
            sum_counts[x2+z] += 1

    # Improved Initialization: Try multiple random orderings
    best_init_A = []
    best_init_size = 0
    
    for _ in range(5):
        A_temp = []
        in_A_temp = set()
        sum_counts_temp = [0] * (2 * N + 1)
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Inline try_add for speed in init
            if x in in_A_temp:
                continue
            if sum_counts_temp[2*x] >= 2:
                continue
            valid = True
            for y in A_temp:
                if sum_counts_temp[x+y] >= 2:
                    valid = False
                    break
            if valid:
                in_A_temp.add(x)
                A_temp.append(x)
                sum_counts_temp[2*x] += 1
                for y in A_temp[:-1]:
                    sum_counts_temp[x+y] += 1
        
        if len(A_temp) > best_init_size:
            best_init_size = len(A_temp)
            best_init_A = A_temp[:]
            
    # Use the best initialization
    A = best_init_A[:]
    in_A = set(A)
    sum_counts = [0] * (2 * N + 1)
    for i in range(len(A)):
        for j in range(i, len(A)):
            sum_counts[A[i] + A[j]] += 1

    best_set = A[:]
    best_size = len(A)
    
    # Local Search
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(A, in_A, sum_counts, x):
                added = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if added:
            continue
        
        # 2. Try swaps (1-for-1)
        swapped = False
        for _ in range(50):
            if not A or not candidates:
                break
            y = random.choice(A)
            x = random.choice(candidates)
            if is_valid_swap(A, in_A, sum_counts, y, x):
                perform_swap(A, in_A, sum_counts, y, x)
                swapped = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if swapped:
            continue

        # 3. Try 1-for-2 moves (Structural Change)
        did_1_for_2 = False
        for _ in range(10):
            if not A or len(candidates) < 2:
                break
            y = random.choice(A)
            idx1 = random.randint(0, len(candidates) - 1)
            idx2 = random.randint(0, len(candidates) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(candidates) - 1)
            
            x1 = candidates[idx1]
            x2 = candidates[idx2]
            
            if is_valid_1_for_2(A, in_A, sum_counts, y, x1, x2):
                perform_1_for_2(A, in_A, sum_counts, y, x1, x2)
                did_1_for_2 = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if did_1_for_2:
            continue

        # 4. If stuck, remove a random element to break deadlock
        if A and random.random() < 0.3:
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
        
    return best_set
