def _greedy_initial():
    S = []
    for x in range(1, 25):
        if _can_add(x, S):
            S.append(x)
    return S

# Modify construct to use greedy init
def construct():
...
