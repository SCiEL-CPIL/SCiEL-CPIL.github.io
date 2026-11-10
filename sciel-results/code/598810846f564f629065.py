def _intersect(p1,p2,p3,p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    d=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if d==0: return None
    n1=x1*y2-y1*x2; n2=x3*y4-y3*x4
    xi=(n1*(x3-x4)-(x1-x2)*n2)/d
    yi=(n1*(y3-y4)-(y1-y2)*n2)/d
    if abs(xi-round(xi))<1e-9 and abs(yi-round(yi))<1e-9:
        xi=int(round(xi)); yi=int(round(yi))
        if -1000<=xi<=1000 and -1000<=yi<=1000:
            return (xi,yi)
    return None
