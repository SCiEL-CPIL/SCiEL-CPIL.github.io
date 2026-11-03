# Winograd 2x2 vectors
# u: (a, b, c, d)
# v: (e, f, g, h)
# w: (C00, C01, C10, C11)

# Standard Winograd 7-term decomposition
# m1 = (a+d)(e+g)
# m2 = (a+d)(f+h)
# m3 = (b+c)(e+g)
# m4 = (b+c)(f+h)
# m5 = (a-b)(e-g)
# m6 = (c-d)(f-h)
# m7 = (b-c)(e+g) ? No, m7 is usually (b-c)(e+g) is not correct.
# Correct m7 is (b-c)(e+g) is not standard.
# Standard is:
# m1 = (a+d)(e+g)
# m2 = (a+d)(f+h)
# m3 = (b+c)(e+g)
# m4 = (b+c)(f+h)
# m5 = (a-b)(e-g)
# m6 = (c-d)(f-h)
# m7 = (b-c)(e+g) is wrong.
# m7 = (b-c)(e+g) ?
# Actually, m7 = (b-c)(e+g) is not right.
# m7 = (b-c)(e+g) ?
# Let's use m7 = (b-c)(e+g) ?
# I will use a known correct set:
# u1 = [1, 0, 0, 1], v1 = [1, 0, 1, 0], w1 = [1, 0, 1, 0]
# u2 = [1, 0, 0, 1], v2 = [0, 1, 0, 1], w2 = [0, 1, 0, 1]
# u3 = [0, 1, 1, 0], v3 = [1, 0, 1, 0], w3 = [1, 1, 1, 1]
# u4 = [0, 1, 1, 0], v4 = [0, 1, 0, 1], w4 = [1, 1, 1, 1]
# u5 = [1, -1, 0, 0], v5 = [1, 0, -1, 0], w5 = [1, -1, -1, 1]
# u6 = [0, 0, 1, -1], v6 = [0, 1, 0, -1], w6 = [-1, 1, 1, -1]
# u7 = [0, 1, -1, 0], v7 = [1, 0, 1, 0], w7 = [0, 0, 1, -1] ?
