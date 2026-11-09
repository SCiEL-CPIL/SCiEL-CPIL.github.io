import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Record start time for deadline guard
    start_time = time.time()
    deadline = 145.0  # 150s limit, 5s safety margin
    
    # The optimal solution for OneMax is a string of N ones.
    # This is a direct construction, so it completes instantly.
    # We implement the deadline guard as requested, although it will never trigger
    # for this specific O(1) operation.
    
    # Direct construction of the global optimum
    best_result = "1" * N
    
    # Check deadline (redundant for this simple task but required by directive)
    if time.time() > deadline:
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
