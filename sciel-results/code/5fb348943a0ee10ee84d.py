"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Construct the standard Černý automaton for N states.
    # This automaton is known to have a shortest reset word of length binom(N-1, 2).
    # For N=12, this length is 55.
    
    # Letter a: cycle (0 -> 1 -> ... -> N-1 -> 0)
    a = [(i + 1) % N for i in range(N)]
    
    # Letter b:
    # b(0) = 0
    # b(i) = i - 1 for i > 0
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    return [a, b]
# EVOLVE-BLOCK-END
