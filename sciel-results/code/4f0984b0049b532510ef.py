import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Helper to check if x can be added to a set represented by in_S
    def can_add(x, in_S):
        # x is 1-indexed
        # Check if x, x+d, x+2d, x+3d in S
        # x is the first term
        for d in range(1, (N - x) // 3 + 1):
            if in_S[x+d] and in_S[x+2*d] and in_S[x+3*d]:
                return False
        # x is the second term: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if in_S[x-d] and in_S[x+d] and in_S[x+2*d]:
                return False
        # x is the third term: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, N-x) + 1):
            if in_S[x-2*d] and in_S[x-d] and in_S[x+d]:
                return False
        # x is the fourth term: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            if in_S[x-3*d] and in_S[x-2*d] and in_S[x-d]:
                return False
        return True

    best_size = 0
    best_set = None
    
    # Randomized Greedy Construction
    # We will run multiple iterations with different random seeds/orderings
    # and keep the best result.
    
    max_iterations = 2000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        order = list(range(1, N+1))
        random.shuffle(order)
        
        in_S = [False] * (N + 1)
        count = 0
        
        for x in order:
            if can_add(x, in_S):
                in_S[x] = True
                count += 1
        
        if count > best_size:
            best_size = count
            best_set = in_S.copy()
            
    # Local Search / Polishing on the best set found
    if best_set is not None:
        # Try to add any remaining elements in random order
        # Repeat this a few times to catch elements that become addable
        # after other elements are added (though in this greedy phase,
        # we only add, so dependencies are one-way. 
        # However, if we did swaps, we might need multiple passes.
        # Here, we just try to add what's left.)
        
        # Actually, the greedy pass already tried all elements in a specific order.
        # If an element was rejected, it was because of the *current* set.
        # As we add more elements, it becomes *harder* to add new ones.
        # So a single pass is optimal for pure addition?
        # No, if A is rejected because of B, and B is not in the set yet?
        # In the loop, we process elements one by one.
        # If x is rejected, it's because of elements already in `in_S`.
        # Elements later in the order are not in `in_S` yet.
        # So if x is rejected due to y, and y is later in the order, x *could* have been added if y wasn't there.
        # But y is not there yet. So x should have been added?
        # Wait. `can_add` checks against `in_S`.
        # If x is rejected, it means there is a 4-AP formed by x and elements *currently* in `in_S`.
        # So x cannot be added regardless of future elements (which would only make it worse).
        # So a single pass is sufficient for pure addition?
        # Yes, because adding elements only increases the constraints.
        
        # However, we might improve by *swapping*.
        # Let's try a simple swap local search.
        # For each x not in S, for each y in S, check if (S - {y}) + {x} is valid.
        # This is expensive: O(N^2 * check_cost).
        # N=512. N^2 ~ 260k. Check cost ~ 700. Total ~ 180M ops.
        # Might be too slow in Python.
        
        # Let's stick to the randomized greedy which is fast.
        # The randomized greedy usually finds sets close to the optimum for this size.
        
        pass

    # Convert best_set to sorted list
    if best_set:
        result = [i for i in range(1, N+1) if best_set[i]]
    else:
        # Fallback to original greedy if something went wrong
        result = []
        in_S = [False] * (N + 1)
        for x in range(1, N+1):
            if can_add(x, in_S):
                in_S[x] = True
                result.append(x)
        
    return result
