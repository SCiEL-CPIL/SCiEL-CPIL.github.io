import numpy as np

def construct():
    N = 512
    # Create an array of numbers 1 to N
    nums = np.arange(1, N + 1)
    
    # We want to count divisors for each number.
    # A number i divides j if j % i == 0.
    # We can create a matrix M where M[i, j] = 1 if (i+1) divides (j+1) ?
    # No, that's N^2 memory. 512x512 is fine.
    
    # Let's use a sieve-like approach with numpy.
    # div_counts = np.zeros(N+1, dtype=int)
    # For i in 1..N:
    #    div_counts[i::i] += 1
    
    div_counts = np.zeros(N + 1, dtype=int)
    for i in range(1, N + 1):
        div_counts[i::i] += 1
        
    # Find the index with the maximum count
    # Note: div_counts[0] is unused.
    max_count = np.max(div_counts[1:])
    best_L = np.argmax(div_counts[1:]) + 1
    
    # Get divisors of best_L
    divs = [i for i in range(1, N + 1) if best_L % i == 0]
    
    return divs
