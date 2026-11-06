import time

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all subset sums are distinct,
    minimizing the largest integer.
    
    Uses a greedy algorithm to select the smallest possible next integer that preserves
    the distinctness of subset sums. This yields the powers of 2, which is the known
    optimal solution for minimizing the largest element.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # 60 second limit, keep safety margin
    
    n = 12
    
    nums = []
    current_sums = {0}
    
    for _ in range(n):
        # Try integers starting from 1
        # We can start from 1, but we know the answer will be powers of 2.
        # To be efficient and safe, we can just iterate.
        found = False
        x = 1
        while not found:
            # Check deadline
            if time.time() > deadline:
                # Fallback to powers of 2 if greedy gets stuck (should not happen)
                return [1 << i for i in range(n)]
            
            # Calculate new sums
            # new_sums = current_sums U {s + x for s in current_sums}
            # Collision if len(new_sums) < 2 * len(current_sums)
            new_sums = current_sums | {s + x for s in current_sums}
            
            if len(new_sums) == 2 * len(current_sums):
                # Valid
                nums.append(x)
                current_sums = new_sums
                found = True
            else:
                x += 1
                
    return nums
