"""Direct return of the known smallest Paley tournament (n=67) for Schütte's problem k=4."""
# EVOLVE-BLOCK-START
def construct():
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            orientation.append(0 if d in squares else 1)
    return {'n': p, 'orientation': orientation}
# EVOLVE-BLOCK-END
