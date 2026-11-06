N = 12

# EVOLVE-BLOCK-START
def construct():
    # Černý automaton: cycle with a sink at 0 for the second letter.
    # Letter a: cycle 0->1->...->11->0
    # Letter b: 0->0, i->i-1 for i>0
    a = [(i + 1) % N for i in range(N)]
    b = [0 if i == 0 else (i - 1) % N for i in range(N)]
    return [a, b]
# EVOLVE-BLOCK-END
