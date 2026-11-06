p = 1009
qr = set((x*x)%p for x in range(1,p))
print(1 in qr, 2 in qr, 3 in qr, 4 in qr, 5 in qr)
