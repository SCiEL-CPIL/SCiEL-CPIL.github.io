317: 
318: # Quick self-test
319: if __name__ == "__main__":
320:     inner, center, side, angle = construct()
321:     print(f"Outer side: {side:.4f}, center: {center}, angle: {angle}")
322:     print(f"Inner hexagons: {len(inner)}")
323:     for i, (x, y, a) in enumerate(inner):
324:         print(f"  Hex {i}: ({x:.3f}, {y:.3f}, {a:.1f} deg)")
325: </search>
326: 
