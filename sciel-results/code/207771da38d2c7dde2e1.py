"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    start_time = time.time()
    
    # Construct two orthogonal Latin squares of order 10
    # Using the decomposition i = 5r + a, j = 5c + b
    # L1 = 5*((r+c)%5) + ((a+b)%5)
    # L2 = 5*((r+2c)%5) + ((a+2b)%5)
    
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        r = i // 5
        a = i % 5
        for j in range(n):
            c = j // 5
            b = j % 5
            
            val1_high = (r + c) % 5
            val1_low = (a + b) % 5
            L1[i][j] = 5 * val1_high + val1_low
            
            val2_high = (r + 2 * c) % 5
            val2_low = (a + 2 * b) % 5
            L2[i][j] = 5 * val2_high + val2_low
            
            # Time guard (though this is O(100), it's negligible)
            if time.time() - start_time > 59:
                break
    
    # Return list of squares
    # Ensure types are Python int
    return [L1, L2]

# EVOLVE-BLOCK-END
