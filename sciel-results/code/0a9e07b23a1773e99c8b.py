import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    base = [1, 2, 5, 11]
    base_set = set(base)
    best_size = len(base)
    best_solution = base[:]

    def compute_sums(lst):
        s = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                s.add(a + lst[j])
        return s

    def is_valid(lst, x, sums):
        # Check if x can be added to lst
        if x in lst: # lst is sorted, but x is new
            return False
        # Check sums
        for y in lst:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_to_sums(sums, lst, x):
        new_sums = set(sums)
        for y in lst:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    def remove_from_sums(sums, lst, x):
        # We don't actually remove, we pass new sets or use a counter dict
        # For simplicity in B&B, passing sets is O(N^2) copy which is slow.
        # Better to use a dictionary of counts for sums.
        pass

    # Use a dictionary for sums to allow O(1) add/remove
    # sum_count[s] = number of pairs (a,b) in current_list such that a+b=s
    
    def backtrack(start_val, current_list, sum_counts, depth):
        nonlocal best_size, best_solution
        if time.time() > deadline:
            return
        
        # Pruning
        # Max elements we can still add: 284 - start_val + 1
        if len(current_list) + (284 - start_val + 1) <= best_size:
            return
            
        # Try adding each x from start_val to 284
        for x in range(start_val, 285):
            # Pruning inside loop
            if len(current_list) + 1 + (284 - x + 1) <= best_size:
                break
                
            # Check validity
            valid = True
            for y in current_list:
                if sum_counts.get(x + y, 0) > 0:
                    valid = False
                    break
            if not valid:
                continue
            if sum_counts.get(2 * x, 0) > 0:
                continue

            # Add x
            current_list.append(x)
            for y in current_list[:-1]: # all y < x
                s = x + y
                sum_counts[s] = sum_counts.get(s, 0) + 1
            s_2x = 2 * x
            sum_counts[s_2x] = sum_counts.get(s_2x, 0) + 1

            # Update best
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_solution = current_list[:]
                
            # Recurse
            backtrack(x + 1, current_list, sum_counts, depth + 1)
            
            # Backtrack
            current_list.pop()
            for y in current_list:
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
            sum_counts[s_2x] -= 1
            if sum_counts[s_2x] == 0:
                del sum_counts[s_2x]

    # Initialize sum_counts for base
    initial_sums = {}
    for i in range(len(base)):
        for j in range(i, len(base)):
            s = base[i] + base[j]
            initial_sums[s] = initial_sums.get(s, 0) + 1
            
    # Start backtracking from the next integer after max(base)
    # Base is sorted: [1, 2, 5, 11]
    # Next start is 12
    backtrack(12, base[:], initial_sums, 0)
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
