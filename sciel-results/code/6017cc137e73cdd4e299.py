m1 = (a00 + a11) * (b00 + b11)
m2 = (a10 + a11) * b00
m3 = a00 * (b10 - b11)
m4 = a11 * (b01 - b00)
m5 = (a00 + a01) * b11
m6 = (a10 - a00) * (b00 + b01)
m7 = (a01 - a11) * (b10 + b11)

c00 = m1 - m5 + m7 + m4  (same as before, gives 15)
