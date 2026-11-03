p1 = (a + b) * (e + f)
p2 = (c + d) * e
p3 = a * (f - h)
p4 = d * (g - e)
p5 = (a + d) * (e + h)
p6 = (b - c) * (g + h)
p7 = (a - c) * (e + f)

c[0][0] = p5 + p3 - p4 - p7
c[0][1] = p1 - p5 - p3 + p6
c[1][0] = p2 + p5 + p4 - p1
c[1][1] = p1 + p7 - p2 - p6
