def neighbours(vertex):
    side, x = vertex
    for first in range(q):
        y = [None, first] + [0] * (d - 1)
        for c in range(2, d + 1):
            a, b = rule[c]
            if side == 0:
                y[c] = add(x[c], mul(y[a], x[b]))
            else:
                y[c] = sub(x[c], mul(x[a], y[b]))
        yield 1 - side, tuple(y)
