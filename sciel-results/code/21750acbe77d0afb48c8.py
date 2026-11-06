N = 18

# EVOLVE-BLOCK-START
def construct():
    # Bubble sort network: N(N-1)/2 comparators.
    # For N=18, this is 153 comparators.
    comparators = []
    for i in range(N - 1):
        for j in range(N - 1 - i):
            comparators.append((j, j + 1))
    return comparators
# EVOLVE-BLOCK-END
