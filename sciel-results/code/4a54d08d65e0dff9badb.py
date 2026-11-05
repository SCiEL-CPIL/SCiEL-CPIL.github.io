"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Known optimal Černý automaton construction (Černý 1964, Pin 1971).
    # For n states, this automaton has reset threshold (n-1)^2.
    #
    # States are 0..n-1.
    # Letter 'a': cyclic permutation  i -> (i+1) % n
    # Letter 'b': 0 -> 0,  i -> i-1 for i >= 1
    #
    # The shortest reset word is a^(n-1) (b a^(n-1))^(n-1), length (n-1)^2.
    # For n=12, that is 121.
    
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return [a, b]
# EVOLVE-BLOCK-END
